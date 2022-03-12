namespace TltfConsole
{
    internal class Config
    {
        public Dictionary<string, string> Settings { get; set; } = new();
        public Config(string path)
        {
            using StreamReader sr = new(path);
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                if (!string.IsNullOrEmpty(line) && line[0] != '#')
                {
                    var value = line.Split('=');
                    if (value.Length == 2)
                    {
                        Settings.Add(value[0].Trim(), value[1].Trim());
                    }
                }
            }
        }
    }
}
