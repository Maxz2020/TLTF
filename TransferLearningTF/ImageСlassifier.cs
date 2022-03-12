using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text;

namespace TransferLearningTF
{
    // https://docs.microsoft.com/ru-ru/dotnet/machine-learning/tutorials/image-classification
    public class ImageСlassifier
    {
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public string[] ImageTypes { get; } = new string[] { ".png", ".jpg", ".jpeg" };

        private readonly double testImagesPart = 0.1;

        private readonly string _assetsPath;
        private readonly string _inceptionTensorFlowModel;
        private readonly string _trainTagsTsv;
        private readonly string _testTagsTsv;
        private string _imagesFolder;

        private readonly MLContext _mlContext = new();
        private ITransformer? _model;
        private DataViewSchema? _schema;
        private IEnumerable<ImagePrediction>? _imagePredictionData;
        private MulticlassClassificationMetrics? _metrics;

        private readonly object lockObj = new();

        /// <summary>
        /// Список имен файлов изобраажений на которы производилось обучение
        /// </summary>
        public List<string?> TestImagesList { get; } = new();

        /// <summary>
        /// Логарифмические потери. Значение логарифмических потерь должно быть максимально близко к нулю.
        /// </summary>
        public double? LogLoss => _metrics?.LogLoss;

        /// <summary>
        /// Значение логарифмических потерь для каждого класса должно быть максимально близко к нулю.
        /// </summary>
        public IReadOnlyCollection<double>? PerClassLogLoss => _metrics?.PerClassLogLoss;

        public ImageСlassifier(string assetsPath)
        {
            _assetsPath = Path.Combine(Environment.CurrentDirectory, assetsPath);
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
            _imagesFolder = Path.Combine(_assetsPath, "images");
            _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
            _testTagsTsv = Path.Combine(_imagesFolder, "tags_test.tsv");
        }

        public void GenerateModel(string imagesFolder)
        {
            _imagesFolder = Path.Combine(_assetsPath, imagesFolder);

            CreateTsvFiles();
            GenerateModel();
            TestModel();
        }

        public void LoadModel(string modelFileName)
        {
            _model = _mlContext.Model.Load(Path.Combine(_assetsPath, modelFileName), out _schema);
        }

        public static string DisplayPredictionData(ImagePrediction prediction)
        {
            StringBuilder stringBuilder = new();
            stringBuilder.AppendLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
            return stringBuilder.ToString();
        }

        public void SaveModel(string modelFileName)
        {
            _mlContext.Model.Save(_model, _schema, Path.Combine(_assetsPath, modelFileName));
        }

        public string DisplayCurrentModelMetric()
        {
            StringBuilder stringBuilder = new();
            if (_imagePredictionData != null)
            {
                foreach (ImagePrediction prediction in _imagePredictionData)
                {
                    stringBuilder.AppendLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
                }
            }

            if (_metrics != null)
            {
                stringBuilder.AppendLine($"LogLoss is: {_metrics.LogLoss}");
                stringBuilder.AppendLine($"PerClassLogLoss is: {string.Join(" , ", _metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            }

            return stringBuilder.ToString();
        }

        public ImagePrediction ClassifySingleImage(string imageFileName)
        {
            if (_model == null)
            {
                throw new InvalidOperationException("Модель не создана.");
            }

            var imageData = new ImageData()
            {
                ImagePath = imageFileName
            };

            //PredictionEngine не является потокобезопасным. Допустимо использовать в средах прототипов или средах с одним потоком.
            lock (lockObj)
            {
                var predictor = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(_model);

                return predictor.Predict(imageData);
            }
        }

        private void GenerateModel()
        {
            IEstimator<ITransformer> pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                // Этот этап конвейера загружает модель TensorFlow в память, а затем обрабатывает вектор значений пикселей через сеть модели TensorFlow. Применение входных данных к модели глубокого обучения и формирование выходных данных с помощью модели называется оценкой. При полном использовании модели оценка делает вывод или прогноз.
                // В этом случае используется вся модель TensorFlow, за исключением последнего слоя, который делает вывод. Выходные данные предпоследнего слоя помечаются как softmax_2_preactivation. Выходные данные этого слоя фактически являются вектором признаков, характеризующих исходные входные изображения.
                // Этот вектор признаков, созданный моделью TensorFlow, будет использоваться в качестве входных данных для алгоритма обучения ML.NET.
                .Append(_mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                // Средство оценки, чтобы сопоставлять строковые метки в данных для обучения с целочисленными значениями ключа
                // Для обучающего алгоритма ML.NET, добавляемого далее, метки должны быть в формате key, а не в виде произвольных строк. Ключ — это число, которое содержит сопоставление по принципу "один к одному" со строковым значением.
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                // Алгоритм обучения ML.NET:
                .Append(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                // Cредство оценки для преобразования значения прогнозируемого ключа обратно в строку:
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(_mlContext);

            IDataView trainingData = _mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
            trainingData = _mlContext.Data.ShuffleRows(trainingData);

            _model = pipeline.Fit(trainingData);
            _schema = trainingData.Schema;
        }

        private void TestModel()
        {
            if (_model != null)
            {
                // Загрузите и преобразуйте проверочные данные
                IDataView testData = _mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
                IDataView predictions = _model.Transform(testData);

                // Create an IEnumerable for the predictions for displaying results
                _imagePredictionData = _mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);

                //Получив прогноз, с помощью метода Evaluate() вы сможете: оценить модель(сравнивает спрогнозированные значения с тестовым набором данных labels); получить метрики производительности модели.
                _metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");
            }
        }

        private void CreateTsvFiles()
        {
            var files = Directory.GetFiles(_imagesFolder, "*", searchOption: SearchOption.AllDirectories);
            var tsv = new List<(string?, string?)>(files.Length);
            int testFilesCount = (int)(Convert.ToDouble(files.Length) * testImagesPart);
            testFilesCount = testFilesCount == 0 ? 1 : testFilesCount;
            var tsvTest = new List<(string?, string?)>(testFilesCount);
            var rnd = new Random();

            foreach (var file in files)
            {
                if (ImageTypes.Contains(Path.GetExtension(file)))
                {
                    var label = Directory.GetParent(file)?.Name;
                    if (label != null)
                    {
                        tsv.Add((file, label));
                    }
                }
            }

            TestImagesList.AddRange(tsv.Select(x => x.Item1));

            for (int i = 0; i < testFilesCount; i++)
            {
                var j = rnd.Next(0, tsv.Count - 1);
                if (!(tsv[j] == (null, null)))
                {
                    tsvTest.Add(tsv[j]);
                    tsv[j] = (null, null);
                }
                else
                {
                    i--;
                }
            }
            tsv = tsv.Where(x => x != (null, null)).ToList();

            using (StreamWriter sw = new(_trainTagsTsv, false, Encoding.Default))
            {
                foreach (var line in tsv)
                {
                    sw.WriteLine($"{line.Item1} \t {line.Item2}");
                }
            }

            using (StreamWriter sw = new(_testTagsTsv, false, Encoding.Default))
            {
                foreach (var line in tsvTest)
                {
                    sw.WriteLine($"{line.Item1} \t {line.Item2}");
                }
            }
        }
    }
}
