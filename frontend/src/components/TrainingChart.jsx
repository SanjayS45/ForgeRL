import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

export function LossChart({ data, title = "Loss" }) {
  return (
    <div className="bg-black/40 rounded-lg p-3">
      <p className="text-[10px] text-emerald-600 mb-2 uppercase">{title}</p>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00ff6a" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00ff6a" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
            <XAxis dataKey="epoch" stroke="#2d4a2d" fontSize={9} />
            <YAxis stroke="#2d4a2d" fontSize={9} />
            <Tooltip
              contentStyle={{
                background: '#0a120a',
                border: '1px solid #00ff6a33',
                borderRadius: 8,
                fontSize: 10,
              }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#00ff6a"
              fill="url(#lossGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function AccuracyChart({ data, title = "Accuracy %" }) {
  return (
    <div className="bg-black/40 rounded-lg p-3">
      <p className="text-[10px] text-emerald-600 mb-2 uppercase">{title}</p>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
            <XAxis dataKey="epoch" stroke="#2d4a2d" fontSize={9} />
            <YAxis stroke="#2d4a2d" fontSize={9} domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                background: '#0a120a',
                border: '1px solid #00ff6a33',
                borderRadius: 8,
                fontSize: 10,
              }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function RewardChart({ data, title = "Episode Reward" }) {
  return (
    <div className="bg-black/40 rounded-lg p-3">
      <p className="text-[10px] text-emerald-600 mb-2 uppercase">{title}</p>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00cc55" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#00cc55" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
            <XAxis dataKey="step" stroke="#2d4a2d" fontSize={9} />
            <YAxis stroke="#2d4a2d" fontSize={9} />
            <Tooltip
              contentStyle={{
                background: '#0a120a',
                border: '1px solid #00ff6a33',
                borderRadius: 8,
                fontSize: 10,
              }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#00cc55"
              fill="url(#rewardGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default { LossChart, AccuracyChart, RewardChart };

