using TransferLearningTF;

namespace TltfConsole
{
    class Program
    {
        private const string AssetsFolderName = "assets";
        private const string ImagesFolderName = "images";
        private const string ResultFolderName = "result";

        private static Logger? _logger;
        private static Config? _config;
        private static List<string?> _testImagesList { get; } = new();

        static void Main(string[] args)
        {
            _config = new(Path.Combine(Environment.CurrentDirectory, AssetsFolderName, "config.txt"));
            _logger = new(Path.Combine(Environment.CurrentDirectory, AssetsFolderName, ResultFolderName, "log.txt"));

            ImageСlassifier imageСlassifier = new(AssetsFolderName);

            _config.Settings.TryGetValue("LoadModel", out var loadModel);

            if (!string.IsNullOrEmpty(loadModel))
            {
                _logger.Log($"Загрузка модели - {loadModel}", true);
                imageСlassifier.LoadModel(loadModel);

                _logger.Log($"Получение списка обучающих файлов...", true);
                var files = Directory.GetFiles(Path.Combine(Environment.CurrentDirectory, AssetsFolderName, ImagesFolderName), "*", searchOption: SearchOption.AllDirectories);
                foreach (var file in files)
                {
                    if (imageСlassifier.ImageTypes.Contains(Path.GetExtension(file)))
                    {
                        _testImagesList.Add(file);
                    }
                }
            }
            else
            {
                _config.Settings.TryGetValue("SaveModel", out var saveModel);
                _config.Settings.TryGetValue("TryCount", out var tryCount);

                _ = int.TryParse(tryCount, out var maxI);
                if (maxI == 0)
                {
                    maxI = 3;
                }
                _logger.Log($"Создание модели c {maxI} попытками...", true);

                double? logLoss = 9;
                for (int i = 1; i <= maxI; i++)
                {
                    _logger.Log($"Попытка {i}", true);
                    imageСlassifier.GenerateModel(ImagesFolderName);

                    if (logLoss > imageСlassifier.LogLoss)
                    {
                        logLoss = imageСlassifier.LogLoss;
                        if (!string.IsNullOrEmpty(saveModel))
                        {
                            _logger.Log($"Сохранение модели - {saveModel} c LogLoss = {logLoss}", true);
                            imageСlassifier.SaveModel(saveModel);
                            _logger.Log(imageСlassifier.DisplayCurrentModelMetric(), true);
                        }
                    }
                }

                _testImagesList.AddRange(imageСlassifier.TestImagesList);
            }

            _config.Settings.TryGetValue("SourseImages", out var value);
            string sourseImages = value ?? ImagesFolderName;

            _config.Settings.TryGetValue("ResultImagesCount", out value);
            _ = int.TryParse(value, out var resultImagesCount);

            _config.Settings.TryGetValue("MaxScoreLimit", out value);
            _ = float.TryParse(value?.Replace('.', ','), out var maxScoreLimit);

            ImageEngine imageEngine = new(Path.Combine(Environment.CurrentDirectory, sourseImages),
                                            Path.Combine(Environment.CurrentDirectory, AssetsFolderName, ResultFolderName),
                                            imageСlassifier, resultImagesCount, maxScoreLimit,
                                            _testImagesList);
            Console.CursorVisible = false;
            imageEngine.StartCopy(GetProgress, GetPredictionInfo);
            Console.CursorVisible = true;
            Console.WriteLine();
            _logger.Log($"Всего файлов:{_totalImages}, обработано:{_processedImages}, найдено:{_copiedImages}");

            Console.WriteLine("Для завершения нажмите любую клавишу...");
            _ = Console.ReadKey();
        }

        private static void GetPredictionInfo(ImagePrediction imagePrediction)
        {
            _logger?.Log(ImageСlassifier.DisplayPredictionData(imagePrediction), false);
        }

        private static int _totalImages;
        private static int _processedImages;
        private static int _copiedImages;

        private static void GetProgress(int totalImages, int processedImages, int copiedImages)
        {
            _totalImages = totalImages;
            _processedImages = processedImages;
            _copiedImages = copiedImages;

            Console.Write($"Всего файлов:{totalImages}, обработано:{processedImages}, скопировано:{copiedImages}");
            Console.SetCursorPosition(0, Console.GetCursorPosition().Top);
        }
    }
}