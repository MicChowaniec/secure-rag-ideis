using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

internal static class Program
{
    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static string FindProjectRoot()
    {
        string current = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        for (int i = 0; i < 3 && current != null; i++)
        {
            if (File.Exists(Path.Combine(current, "pyproject.toml"))) return current;
            DirectoryInfo parent = Directory.GetParent(current);
            current = parent == null ? null : parent.FullName;
        }
        return null;
    }

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            Console.OutputEncoding = new UTF8Encoding(false);
            Console.Title = "Bezpieczny Chatbot RAG IDEIS";
            string root = FindProjectRoot();
            if (root == null)
            {
                Console.Error.WriteLine("Nie znaleziono pyproject.toml. Umieść EXE w rozpakowanym folderze projektu.");
                Console.ReadLine();
                return 2;
            }

            string tempDir = Path.Combine(Path.GetTempPath(), "SecureRagBot");
            Directory.CreateDirectory(tempDir);
            string scriptPath = Path.Combine(tempDir, "bootstrap.ps1");
            using (Stream source = Assembly.GetExecutingAssembly().GetManifestResourceStream("bootstrap.ps1"))
            {
                if (source == null) throw new InvalidOperationException("Brak osadzonego skryptu bootstrap.ps1.");
                using (var reader = new StreamReader(source, new UTF8Encoding(false), true))
                {
                    // Windows PowerShell 5.1 interprets UTF-8 without BOM as the legacy
                    // system code page. Write a BOM so Polish text cannot break parsing.
                    File.WriteAllText(scriptPath, reader.ReadToEnd(), new UTF8Encoding(true));
                }
            }

            bool selfTest = args != null && Array.Exists(args,
                value => string.Equals(value, "--self-test", StringComparison.OrdinalIgnoreCase));
            var start = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = "-NoProfile -ExecutionPolicy Bypass -File " + Quote(scriptPath) +
                            " -ProjectRoot " + Quote(root) + (selfTest ? " -SelfTest" : ""),
                UseShellExecute = false,
                CreateNoWindow = false,
                WorkingDirectory = root
            };
            using (Process process = Process.Start(start))
            {
                process.WaitForExit();
                int exitCode = process.ExitCode;
                if (exitCode != 0 && !selfTest)
                {
                    Console.Error.WriteLine("Instalator zakończył się błędem (kod " + exitCode + ").");
                    Console.Error.WriteLine("Okno pozostanie otwarte. Naciśnij Enter, aby zamknąć.");
                    Console.ReadLine();
                }
                return exitCode;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("BŁĄD URUCHAMIANIA: " + ex.Message);
            Console.Error.WriteLine("Naciśnij Enter, aby zamknąć.");
            Console.ReadLine();
            return 1;
        }
    }
}
