import React, { useState } from 'react';
import type { Pyodide } from '../types';
import { PlayIcon, TerminalIcon } from '../constants';

interface CodeCellProps {
  code: string;
  pyodide: Pyodide | null;
}

export const CodeCell: React.FC<CodeCellProps> = ({ code, pyodide }) => {
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const runCode = async () => {
    if (!pyodide) {
      setError("Python environment not ready. Please wait.");
      return;
    }
    setIsRunning(true);
    setOutput('');
    setError('');

    // Capture stdout and stderr
    const stdout: string[] = [];
    const stderr: string[] = [];
    pyodide.globals.set('stdout_callback', (s: string) => stdout.push(s));
    pyodide.globals.set('stderr_callback', (s: string) => stderr.push(s));
    
    const pythonSetupCode = `
import sys
import io

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
    `;

    const pythonCaptureCode = `
sys.stdout.seek(0)
sys.stderr.seek(0)
stdout_callback(sys.stdout.read())
stderr_callback(sys.stderr.read())
    `

    try {
        await pyodide.runPythonAsync(pythonSetupCode);
        await pyodide.runPythonAsync(code);
    } catch (e: any) {
        setError(e.message);
    } finally {
        await pyodide.runPythonAsync(pythonCaptureCode);
        setOutput(stdout.join('\n'));
        if (stderr.length > 0) {
            setError(prev => prev ? prev + '\n' + stderr.join('\n') : stderr.join('\n'));
        }
        setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-md my-4 overflow-hidden">
      <div className="p-4 bg-slate-800">
        <pre className="text-sm font-mono text-slate-300 whitespace-pre-wrap overflow-x-auto">
          <code>{code}</code>
        </pre>
      </div>
      <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700 flex items-center">
        <button
          onClick={runCode}
          disabled={isRunning || !pyodide}
          className="flex items-center px-4 py-2 text-sm font-semibold text-white bg-cyan-600 rounded-md hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
              Running...
            </>
          ) : (
            <>
              <PlayIcon />
              <span className="ml-2">Run Cell</span>
            </>
          )}
        </button>
      </div>
      {(output || error) && (
        <div className="px-4 py-3 bg-black/30 border-t border-slate-700">
            <h4 className="flex items-center text-xs font-semibold text-slate-400 uppercase mb-2">
                <TerminalIcon />
                <span className="ml-2">Output</span>
            </h4>
          {output && (
            <pre className="text-sm font-mono text-slate-300 whitespace-pre-wrap">{output}</pre>
          )}
          {error && (
            <pre className="text-sm font-mono text-red-400 whitespace-pre-wrap">{error}</pre>
          )}
        </div>
      )}
    </div>
  );
};
