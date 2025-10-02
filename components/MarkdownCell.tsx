import React from 'react';

interface MarkdownCellProps {
  content: {
    title: string;
    level: 'h1' | 'h2' | 'h3';
    description: string;
  };
}

export const MarkdownCell: React.FC<MarkdownCellProps> = ({ content }) => {
  const { title, level, description } = content;

  const renderTitle = () => {
    const commonClasses = "font-bold text-white";
    if (level === 'h1') return <h1 className={`text-3xl sm:text-4xl pb-2 mb-4 border-b-2 border-slate-700 ${commonClasses}`}>{title}</h1>;
    if (level === 'h2') return <h2 className={`text-2xl sm:text-3xl pb-2 mb-4 border-b border-slate-700 ${commonClasses}`}>{title}</h2>;
    if (level === 'h3') return <h3 className={`text-xl sm:text-2xl mb-3 text-cyan-400 ${commonClasses}`}>{title}</h3>;
    return null;
  };

  return (
    <div className="w-full bg-slate-800/30 rounded-lg p-6 my-4">
      {renderTitle()}
      <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{description}</p>
    </div>
  );
};
