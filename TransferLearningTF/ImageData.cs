using Microsoft.ML.Data;

namespace TransferLearningTF
{
    // Класс входных данных изображения 
    public class ImageData
    {
        // Имя файла изображения.
        [LoadColumn(0)]
        public string? ImagePath { get; set; }

        //Значение для метки изображения.
        [LoadColumn(1)]
        public string? Label { get; set; }
    }
}
