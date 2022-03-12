using TransferLearningTF;

namespace TltfConsole
{
    internal class ImageEngine
    {
        public string SoursePath { get; }
        public string ResultPath { get; }
        public int Limit { get; }
        public float MaxScoreLimit { get; }
        public List<string?> TestImagesList { get; } = new();

        private readonly ImageСlassifier _imageСlassifier;

        public ImageEngine(string soursePath, string resultPath, ImageСlassifier imageСlassifier, int limit, float maxScoreLimit, List<string?> testImagesList)
        {
            SoursePath = soursePath;
            ResultPath = resultPath;
            Limit = limit;
            if (maxScoreLimit <= 0 || maxScoreLimit >= 1)
            {
                throw new ArgumentException($"Параметр MaxScoreLimit должен быть больше 0 и меньше 1");
            }
            MaxScoreLimit = maxScoreLimit;
            TestImagesList.AddRange(testImagesList);
            _imageСlassifier = imageСlassifier;
        }

        /// <summary>
        /// Запуск процесса поиска и копирования изображений по заданным параметрам
        /// </summary>
        /// <param name="stausCallback"></param>
        /// <param name="predictionInfoCallback"></param>
        public void StartCopy(Action<int, int, int>? stausCallback = null, Action<ImagePrediction>? predictionInfoCallback = null)
        {
            var files = Directory.GetFiles(SoursePath, "*", searchOption: SearchOption.AllDirectories);
            List<string> imagesList = new();

            var imgTypes = _imageСlassifier.ImageTypes;

            foreach (var file in files)
            {
                if (imgTypes.Contains(Path.GetExtension(file)))
                {
                    imagesList.Add((file));
                }
            }

            var totalImages = imagesList.Count;
            int processedImages = 0;
            int copiedImages = 0;
            foreach (var image in imagesList)
            {
                processedImages++;
                if (processedImages % 10 == 0)
                {
                    stausCallback?.Invoke(totalImages, processedImages, copiedImages);
                }

                ImagePrediction? prediction = null;
                try
                {
                    prediction = _imageСlassifier.ClassifySingleImage(image);
                }
                catch
                {

                }

                if (prediction?.PredictedLabelValue != null && prediction.Score?.Max() > MaxScoreLimit)
                {
                    if (prediction.PredictedLabelValue[0] != '_')
                    {
                        copiedImages++;
                    }
                    FileInfo fileInfo = new(image);

                    if (!Directory.Exists(Path.Combine(ResultPath, prediction.PredictedLabelValue)))
                    {
                        Directory.CreateDirectory(Path.Combine(ResultPath, prediction.PredictedLabelValue));
                    }

                    var newImageName = GetNewImageName(Path.Combine(ResultPath, prediction.PredictedLabelValue, fileInfo.Name), fileInfo.Length);
                    _ = fileInfo.CopyTo(newImageName, true);

                    predictionInfoCallback?.Invoke(prediction);

                    if (copiedImages >= Limit)
                    {
                        break;
                    }
                }
            }
            stausCallback?.Invoke(totalImages, processedImages, copiedImages);
        }

        private string GetNewImageName(string imgFullName, long imgFileLength)
        {
            var imgNewName = imgFullName;

            if (!string.IsNullOrEmpty(imgFullName))
            {
                FileInfo imgFileInfo = new(imgFullName);

                var matchingFiles = TestImagesList.Where(x => Path.GetFileName(x) == Path.GetFileName(imgFullName));

                foreach (var matchedFile in matchingFiles)
                {
                    if (matchedFile != null)
                    {
                        FileInfo matchedFileInfo = new(matchedFile);

                        if (matchedFileInfo.Length == imgFileLength)
                        {
                            if (Directory.GetParent(matchedFile)?.Name == Directory.GetParent(imgFullName)?.Name)
                            {
                                var x = Path.GetDirectoryName(imgFullName);
                                var y = Path.GetFileName(imgFullName);

                                imgNewName = Path.Combine(Path.GetDirectoryName(imgFullName) ?? "", "known_" + Path.GetFileName(imgFullName));
                            }
                        }
                    }
                }
            }

            return imgNewName;
        }
    }
}
