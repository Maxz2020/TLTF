namespace TransferLearningTF
{
    // Прогноз изображения
    public class ImagePrediction : ImageData
    {
        // Процентное значение достоверности для конкретной классификации изображения.
        public float[]? Score { get; set; }

        // Значение для прогнозируемой метки классификации изображения.
        public string? PredictedLabelValue { get; set; }
    }
}
