export interface MarkdownCellData {
  type: 'markdown';
  content: {
    title: string;
    level: 'h1' | 'h2' | 'h3';
    description: string;
  };
}

export interface CodeCellData {
  type: 'code';
  content: string;
}

export type Cell = MarkdownCellData | CodeCellData;

// --- Pyodide Types ---
// These are simplified types for Pyodide interaction

export interface Pyodide {
  runPythonAsync: (code: string, options?: { globals: any }) => Promise<any>;
  loadPackage: (packages: string | string[]) => Promise<void>;
  pyimport: (module: string) => any; // Returns a PyProxy for the imported module
  globals: Map<string, any>;
}
