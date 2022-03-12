using System.Text;

namespace TltfConsole
{
    internal class Logger
    {
        public string FileName { get; }
        public Logger(string fileName)
        {
            FileName = fileName;
            using StreamWriter sw = new(FileName, false, Encoding.Default);
        }

        public void Log(string text, bool echoToConsole = false)
        {
            using StreamWriter sw = new(FileName, true, Encoding.Default);

            sw.WriteLine($"{DateTime.Now}\n{text}");

            if (echoToConsole)
            {
                Console.WriteLine($"{DateTime.Now}\n{text}");
            }
        }
    }
}
