import { GlossaChip } from "./GlossaChip";

type Props = {
  glossas: string[];
  translation: string | null;
  isTranslating: boolean;
  onTranslate: () => void;
  onClear: () => void;
};

export function SidePanel({ glossas, translation, isTranslating, onTranslate, onClear }: Props) {
  return (
    <div className="w-[340px] bg-[#14141e] border-l border-[#2a2a3a] flex flex-col overflow-hidden flex-shrink-0">

      {/* Glossas */}
      <div className="p-4 border-b border-[#2a2a3a]">
        <p className="text-[0.65rem] uppercase tracking-[1.5px] text-gray-500 mb-2">
          Sinais detectados
        </p>
        <div className="flex flex-wrap gap-1.5 min-h-[50px] content-start">
          {glossas.map((g, i) => (
            <GlossaChip key={i} label={g} />
          ))}
        </div>
      </div>

      {/* Translation */}
      <div className="flex-1 overflow-y-auto p-4">
        {isTranslating ? (
          <p className="text-gray-400 flex items-center gap-2 text-sm">
            <Spinner />
            Traduzindo...
          </p>
        ) : translation != null ? (
          <p className="text-[#d4d4d4] leading-relaxed">{translation}</p>
        ) : (
          <p className="text-gray-600 italic text-sm">
            Faça sinais e pressione{" "}
            <strong className="not-italic text-gray-500">Traduzir</strong>.
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="p-3 border-t border-[#2a2a3a] flex gap-2">
        <button
          onClick={onTranslate}
          disabled={glossas.length === 0 || isTranslating}
          className="flex-1 py-2.5 rounded-lg font-semibold text-sm text-white
            bg-gradient-to-br from-indigo-500 to-indigo-400
            disabled:opacity-40 disabled:cursor-not-allowed
            active:scale-[0.97] transition-all cursor-pointer"
        >
          Traduzir
        </button>
        <button
          onClick={onClear}
          className="flex-1 py-2.5 rounded-lg font-semibold text-sm text-gray-400
            bg-[#1e1e30] border border-[#3a3a5a]
            active:scale-[0.97] transition-all cursor-pointer"
        >
          Limpar
        </button>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-gray-600
      border-t-indigo-400 animate-spin flex-shrink-0" />
  );
}
