export function GlossaChip({ label }: { label: string }) {
  return (
    <span className="bg-[#1e1e30] border border-[#3a3a5a] rounded px-2.5 py-1
      text-xs font-semibold text-indigo-300 animate-pop">
      {label}
    </span>
  );
}
