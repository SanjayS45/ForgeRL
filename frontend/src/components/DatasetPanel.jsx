import React from 'react';
import { Upload, Database, Download } from 'lucide-react';

export function DatasetPanel({
  datasets,
  selectedDataset,
  onSelectDataset,
  onUpload,
  datasetDetails,
  apiBase,
}) {
  return (
    <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
      <h2 className="text-sm font-semibold mb-3 flex items-center gap-2 text-neon-green">
        <Database size={16} /> DATASET
      </h2>

      <label className="block w-full p-3 border border-dashed border-neon-green/30 rounded-lg hover:border-neon-green/60 transition cursor-pointer mb-3 bg-black/30">
        <div className="text-center">
          <Upload className="mx-auto mb-1 text-emerald-600" size={20} />
          <p className="text-xs text-emerald-600">Upload .pkl / .csv</p>
        </div>
        <input
          type="file"
          accept=".pkl,.pickle,.csv"
          onChange={onUpload}
          className="hidden"
        />
      </label>

      <div className="space-y-1 max-h-40 overflow-y-auto">
        {datasets.map((ds) => (
          <div
            key={ds.name}
            className={`w-full p-2 rounded-lg text-xs transition flex items-center justify-between ${
              selectedDataset === ds.name
                ? 'bg-neon-green/20 border border-neon-green/40 text-neon-green'
                : 'bg-black/30 hover:bg-black/50 border border-transparent text-emerald-400'
            }`}
          >
            <button
              onClick={() => onSelectDataset(ds)}
              className="text-left flex-1"
            >
              <p className="font-medium">{ds.name}</p>
              <p className="text-emerald-700">
                {ds.num_pairs?.toLocaleString()} pairs
              </p>
            </button>
            <a
              href={`${apiBase}/datasets/${ds.name}/download`}
              className="p-1 hover:bg-neon-green/20 rounded"
              title="Download dataset"
              onClick={(e) => e.stopPropagation()}
            >
              <Download size={12} />
            </a>
          </div>
        ))}
      </div>

      {datasetDetails && (
        <div className="mt-3 p-2 bg-black/40 rounded-lg text-[10px] grid grid-cols-2 gap-1 text-emerald-600">
          <span>Instructions: {datasetDetails.num_instructions}</span>
          <span>Obs: {datasetDetails.obs_dim}D</span>
          <span>Act: {datasetDetails.act_dim}D</span>
          <span>
            Size: {datasetDetails.path ? 'Available' : 'N/A'}
          </span>
        </div>
      )}
    </div>
  );
}

export default DatasetPanel;

