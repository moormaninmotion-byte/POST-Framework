import React, { useState, useEffect, useRef } from 'react';
import { NOTEBOOK_DATA } from './constants';
import type { Cell, Pyodide } from './types';
import { MarkdownCell } from './components/MarkdownCell';
import { CodeCell } from './components/CodeCell';

// Make pyodide globally accessible for the script
declare global {
  interface Window {
    loadPyodide: (config: { indexURL: string }) => Promise<Pyodide>;
  }
}

const PYTHON_MOCK_SETUP_CODE = `
import sys
from collections import OrderedDict

print("Setting up mock simulation environment...")

class MockTensor:
    def __init__(self, *shape):
        self.shape = shape
    
    @property
    def data(self):
        return self
    
    def copy_(self, other):
        pass

    def item(self):
        return 0.12345
    
    def __getitem__(self, key):
        return [0.123, 0.456, 0.789]

    def __str__(self):
        return f"MockTensor(shape={self.shape})"
        
    def __add__(self, other): return MockTensor(self.shape)
    def __sub__(self, other): return MockTensor(self.shape)

class MockModule:
    def __init__(self):
        self._modules = OrderedDict()
        self._params = OrderedDict()

    def __setattr__(self, key, value):
        if isinstance(value, MockModule):
            self._modules[key] = value
        elif isinstance(value, MockTensor):
            self._params[key] = value
        else:
            super().__setattr__(key, value)
    
    def __getattr__(self, key):
        if key in self._modules:
            return self._modules[key]
        if key in self._params:
            return self._params[key]
        # Allow accessing nested modules like .transformer.layer_0
        parts = key.split('.')
        if len(parts) > 1 and parts[0] in self._modules:
            module = self._modules[parts[0]]
            for part in parts[1:]:
                module = getattr(module, part)
            return module
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def state_dict(self):
        return OrderedDict([
            ('transformer.layer_0.weight', MockTensor(10, 10)),
            ('lm_head.weight', MockTensor(10, 5))
        ])

    def load_state_dict(self, state_dict): pass
    def parameters(self): return []


class MockLinear(MockModule):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = MockTensor(out_f, in_f)

class MockSequential(MockModule):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[key] = module

# --- Create torch module ---
torch = type(sys)('torch')
torch.randn = lambda *shape: MockTensor(*shape)

# --- Create torch.nn module ---
nn = type(sys)('torch.nn')
nn.Module = MockModule
nn.Linear = MockLinear
nn.Sequential = MockSequential
torch.nn = nn

# --- Create torch.nn.functional module ---
F = type(sys)('torch.nn.functional')
F.kl_div = lambda i1, i2, reduction: MockTensor()
F.log_softmax = lambda i, dim: MockTensor(i.shape)
F.softmax = lambda i, dim: MockTensor(i.shape)
nn.functional = F

# --- Create torch.optim module ---
optim = type(sys)('torch.optim')
class Adam:
    def __init__(self, params): pass
optim.Adam = Adam
torch.optim = optim

# --- Inject into sys.modules ---
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.nn.functional'] = torch.nn.functional
sys.modules['torch.optim'] = torch.optim

print("Mock 'torch' module ready.")
`;


const App: React.FC = () => {
  const [pyodide, setPyodide] = useState<Pyodide | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState("Initializing Python Environment...");
  const pyodideRef = useRef<Pyodide | null>(null);

  useEffect(() => {
    const initPyodide = async () => {
      try {
        const pyodideInstance = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/",
        });

        setLoadingMessage("Setting up simulation environment...");
        // Set up the mock torch library before any cells are run
        await pyodideInstance.runPythonAsync(PYTHON_MOCK_SETUP_CODE);
        
        pyodideRef.current = pyodideInstance;
        setPyodide(pyodideInstance);
      } catch (error) {
        console.error("Failed to initialize Pyodide:", error);
        setLoadingMessage("Error: Could not initialize Python environment.");
      } finally {
        setIsLoading(false);
      }
    };
    initPyodide();
  }, []);
  
  const renderCell = (cell: Cell, index: number) => {
    if (cell.type === 'markdown') {
      return <MarkdownCell key={index} content={cell.content} />;
    }
    if (cell.type === 'code') {
      return <CodeCell key={index} code={cell.content} pyodide={pyodide} />;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-slate-900 font-sans p-4 sm:p-6 lg:p-8">
       {isLoading && (
        <div className="fixed inset-0 bg-slate-900 bg-opacity-80 flex flex-col items-center justify-center z-50 backdrop-blur-sm">
          <div className="w-16 h-16 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-4 text-lg text-slate-300">{loadingMessage}</p>
        </div>
      )}
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-3">
            POST Framework Interactive Notebook
          </h1>
          <p className="text-lg text-cyan-400">
            Run live Python simulations in your browser to understand Privacy-Preserving Soft-Prompt Transfer.
          </p>
        </header>

        <main className="space-y-6">
          {NOTEBOOK_DATA.map(renderCell)}
        </main>

        <footer className="text-center mt-16 py-6 border-t border-slate-800">
          <p className="text-slate-500">
            Expert Explanation by an Applied Machine Learning AI Assistant
          </p>
        </footer>
      </div>
    </div>
  );
};

export default App;