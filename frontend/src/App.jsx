import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, ScatterChart, Scatter
} from 'recharts';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Cylinder, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import {
  Upload, Play, Pause, Settings, Download, Brain, Bot, Activity,
  RefreshCw, Layers, Eye, Zap, Database, BarChart3, Sliders, Save,
  Target, Plus, Trash2, X
} from 'lucide-react';
import axios from 'axios';

const API_BASE = '/api';

// ==================== PARAMETER INPUT COMPONENT (OUTSIDE App to prevent re-render) ====================

const ParamInput = React.memo(({ label, value, onChange, type = 'number', step, options, min, max, placeholder }) => (
  <div className="space-y-1">
    <label className="text-xs text-emerald-600 uppercase tracking-wider">{label}</label>
    {options ? (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-black/50 border border-neon-green/20 rounded-lg px-3 py-2 text-sm text-emerald-100 focus:border-neon-green focus:outline-none"
      >
        {options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    ) : (
      <input
        type={type}
        step={step}
        min={min}
        max={max}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(type === 'number' ? (e.target.value === '' ? '' : parseFloat(e.target.value)) : e.target.value)}
        className="w-full bg-black/50 border border-neon-green/20 rounded-lg px-3 py-2 text-sm text-emerald-100 focus:border-neon-green focus:outline-none"
      />
    )}
  </div>
));

// Available MetaWorld tasks
const METAWORLD_TASKS = [
  'reach-v3',
  'push-v3', 
  'pick-place-v3',
  'door-open-v3',
  'drawer-open-v3',
  'drawer-close-v3',
  'button-press-v3',
  'window-open-v3',
  'window-close-v3',
  'faucet-open-v3',
  'faucet-close-v3',
  'coffee-push-v3',
  'coffee-pull-v3',
  'coffee-button-v3',
];

// Task descriptions for display
const TASK_DESCRIPTIONS = {
  'reach-v3': 'Move end effector to reach the target position',
  'push-v3': 'Push the puck to the goal position',
  'pick-place-v3': 'Pick up the object and place it at the goal',
  'door-open-v3': 'Open the door by rotating the handle',
  'drawer-open-v3': 'Pull the drawer open',
  'drawer-close-v3': 'Push the drawer closed',
  'button-press-v3': 'Press the button down',
  'window-open-v3': 'Slide the window open',
  'window-close-v3': 'Slide the window closed',
  'faucet-open-v3': 'Turn the faucet to open',
  'faucet-close-v3': 'Turn the faucet to close',
  'coffee-push-v3': 'Push the coffee mug to the target',
  'coffee-pull-v3': 'Pull the coffee mug to the target',
  'coffee-button-v3': 'Press the coffee machine button',
};

// ==================== SAWYER ARM VISUALIZATION ====================

// Inverse Kinematics solver for 2-link arm
function solveIK(targetX, targetY, targetZ, L1, L2) {
  // Project to horizontal plane for base rotation
  const baseAngle = Math.atan2(targetZ, targetX);
  
  // Distance in horizontal plane from base
  const horizontalDist = Math.sqrt(targetX * targetX + targetZ * targetZ);
  
  // 2D IK in the vertical plane (horizontal dist, height)
  const dx = horizontalDist;
  const dy = targetY - 0.25; // Subtract base height
  const dist = Math.sqrt(dx * dx + dy * dy);
  
  // Clamp to reachable workspace
  const maxReach = L1 + L2 - 0.01;
  const minReach = Math.abs(L1 - L2) + 0.01;
  const clampedDist = Math.max(minReach, Math.min(maxReach, dist));
  
  // Law of cosines for elbow angle
  const cosElbow = (L1 * L1 + L2 * L2 - clampedDist * clampedDist) / (2 * L1 * L2);
  const elbowAngle = Math.PI - Math.acos(Math.max(-1, Math.min(1, cosElbow)));
  
  // Shoulder angle
  const cosAlpha = (L1 * L1 + clampedDist * clampedDist - L2 * L2) / (2 * L1 * clampedDist);
  const alpha = Math.acos(Math.max(-1, Math.min(1, cosAlpha)));
  const beta = Math.atan2(dy, dx);
  const shoulderAngle = beta + alpha;
  
  return { baseAngle, shoulderAngle, elbowAngle };
}

function SawyerArm({ trajectory, currentStep, showPath, customTargets = [] }) {
  const armRef = useRef();
  const pulseRef = useRef(0);
  
  // Arm segment lengths
  const L1 = 0.35; // Upper arm
  const L2 = 0.30; // Forearm
  const L3 = 0.15; // Wrist to gripper
  
  // Extract current position from trajectory
  const positions = useMemo(() => {
    return trajectory.map(step => {
      const obs = step[0];
      // Scale positions for better visualization
      return new THREE.Vector3(
        obs[0] * 2.5,
        obs[2] * 2.5 + 0.4,
        obs[1] * 2.5
      );
    });
  }, [trajectory]);
  
  const currentPos = positions[currentStep] || new THREE.Vector3(0.3, 0.5, 0);
  
  // Goal position from trajectory data
  const goalPos = useMemo(() => {
    if (trajectory[currentStep]) {
      const obs = trajectory[currentStep][0];
      // Goal is typically in the last 3 elements of observation
      return new THREE.Vector3(
        obs.slice(-3)[0] * 2.5,
        obs.slice(-3)[2] * 2.5 + 0.4,
        obs.slice(-3)[1] * 2.5
      );
    }
    return new THREE.Vector3(0.4, 0.4, 0.2);
  }, [trajectory, currentStep]);
  
  // Calculate distance to goal for success visualization
  const distanceToGoal = currentPos.distanceTo(goalPos);
  const isNearGoal = distanceToGoal < 0.1;

  // Solve IK for current end effector position
  const { baseAngle, shoulderAngle, elbowAngle } = useMemo(() => {
    return solveIK(currentPos.x, currentPos.y, currentPos.z, L1, L2);
  }, [currentPos, L1, L2]);

  // Animate pulse effect
  useFrame((state) => {
    pulseRef.current = Math.sin(state.clock.elapsedTime * 3) * 0.5 + 0.5;
  });

  // Calculate actual end effector position from forward kinematics (for verification line)
  const fkEndPos = useMemo(() => {
    const s1 = Math.sin(shoulderAngle);
    const c1 = Math.cos(shoulderAngle);
    const s2 = Math.sin(shoulderAngle - elbowAngle);
    const c2 = Math.cos(shoulderAngle - elbowAngle);
    
    const y = 0.25 + L1 * s1 + L2 * s2;
    const r = L1 * c1 + L2 * c2;
    const x = r * Math.cos(baseAngle);
    const z = r * Math.sin(baseAngle);
    
    return new THREE.Vector3(x, y, z);
  }, [baseAngle, shoulderAngle, elbowAngle, L1, L2]);

  return (
    <group ref={armRef}>
      {/* Ground/Table */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[4, 4]} />
        <meshStandardMaterial color="#0a140a" metalness={0.3} roughness={0.8} />
      </mesh>
      
      {/* Grid on ground */}
      <gridHelper args={[4, 20, '#004422', '#001a0a']} position={[0, 0.001, 0]} />
      
      {/* Workspace boundary circle */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[L1 + L2 - 0.05, L1 + L2, 64]} />
        <meshBasicMaterial color="#003300" transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      
      {/* Robot Base */}
      <group position={[0, 0, 0]}>
        {/* Base plate */}
        <Cylinder args={[0.2, 0.22, 0.06, 32]} position={[0, 0.03, 0]} castShadow>
          <meshStandardMaterial color="#1a1a1a" metalness={0.9} roughness={0.2} />
        </Cylinder>
        
        {/* Base column */}
        <Cylinder args={[0.12, 0.14, 0.12, 32]} position={[0, 0.1, 0]} castShadow>
          <meshStandardMaterial color="#222222" metalness={0.8} roughness={0.3} />
        </Cylinder>
        
        {/* Base accent ring */}
        <mesh position={[0, 0.1, 0]}>
          <torusGeometry args={[0.14, 0.008, 8, 32]} />
          <meshStandardMaterial color="#00ff6a" emissive="#00ff6a" emissiveIntensity={0.5} />
        </mesh>
        
        {/* Shoulder mount */}
        <Cylinder args={[0.08, 0.1, 0.1, 32]} position={[0, 0.2, 0]} castShadow>
          <meshStandardMaterial color="#2a2a2a" metalness={0.7} roughness={0.3} />
        </Cylinder>
      </group>
      
      {/* Arm Assembly - Properly connected with IK */}
      <group position={[0, 0.25, 0]} rotation={[0, baseAngle, 0]}>
        {/* Shoulder Joint */}
        <Sphere args={[0.06, 16, 16]} castShadow>
          <meshStandardMaterial color="#2a2a2a" metalness={0.8} roughness={0.2} />
        </Sphere>
        <mesh rotation={[0, 0, Math.PI/2]}>
          <torusGeometry args={[0.07, 0.006, 8, 32]} />
          <meshStandardMaterial color="#00cc55" emissive="#00cc55" emissiveIntensity={0.4} />
        </mesh>
        
        {/* Upper Arm (Link 1) */}
        <group rotation={[0, 0, Math.PI/2 - shoulderAngle]}>
          <Cylinder args={[0.045, 0.04, L1, 16]} position={[L1/2, 0, 0]} rotation={[0, 0, Math.PI/2]} castShadow>
            <meshStandardMaterial color="#333333" metalness={0.6} roughness={0.4} />
          </Cylinder>
          
          {/* Elbow Joint */}
          <group position={[L1, 0, 0]}>
            <Sphere args={[0.05, 16, 16]} castShadow>
              <meshStandardMaterial color="#2a2a2a" metalness={0.8} roughness={0.2} />
            </Sphere>
            <mesh rotation={[0, 0, Math.PI/2]}>
              <torusGeometry args={[0.06, 0.005, 8, 32]} />
              <meshStandardMaterial color="#00cc55" emissive="#00cc55" emissiveIntensity={0.3} />
            </mesh>
            
            {/* Forearm (Link 2) */}
            <group rotation={[0, 0, elbowAngle]}>
              <Cylinder args={[0.035, 0.03, L2, 16]} position={[L2/2, 0, 0]} rotation={[0, 0, Math.PI/2]} castShadow>
                <meshStandardMaterial color="#333333" metalness={0.6} roughness={0.4} />
              </Cylinder>
              
              {/* Wrist Joint */}
              <group position={[L2, 0, 0]}>
                <Sphere args={[0.04, 16, 16]} castShadow>
                  <meshStandardMaterial color="#2a2a2a" metalness={0.8} roughness={0.2} />
                </Sphere>
                
                {/* Wrist segment */}
                <Cylinder args={[0.025, 0.02, L3, 16]} position={[L3/2, 0, 0]} rotation={[0, 0, Math.PI/2]} castShadow>
                  <meshStandardMaterial color="#333333" metalness={0.6} roughness={0.4} />
                </Cylinder>
                
                {/* Gripper Assembly */}
                <group position={[L3, 0, 0]} rotation={[0, 0, -Math.PI/2 + shoulderAngle - elbowAngle]}>
                  {/* Gripper mount */}
                  <Cylinder args={[0.025, 0.03, 0.04, 16]} castShadow>
                    <meshStandardMaterial color="#222222" metalness={0.8} roughness={0.2} />
                  </Cylinder>
                  
                  {/* Gripper base */}
                  <Box args={[0.06, 0.02, 0.03]} position={[0, -0.03, 0]} castShadow>
                    <meshStandardMaterial color="#1a1a1a" metalness={0.7} roughness={0.3} />
                  </Box>
                  
                  {/* Gripper fingers */}
                  <Box args={[0.012, 0.045, 0.015]} position={[0.02, -0.055, 0]} castShadow>
                    <meshStandardMaterial color="#333333" metalness={0.6} roughness={0.4} />
                  </Box>
                  <Box args={[0.012, 0.045, 0.015]} position={[-0.02, -0.055, 0]} castShadow>
                    <meshStandardMaterial color="#333333" metalness={0.6} roughness={0.4} />
                  </Box>
                  
                  {/* Finger tips */}
                  <Box args={[0.015, 0.012, 0.018]} position={[0.02, -0.08, 0]}>
                    <meshStandardMaterial 
                      color={isNearGoal ? "#00ff6a" : "#00aa44"} 
                      emissive={isNearGoal ? "#00ff6a" : "#004422"} 
                      emissiveIntensity={isNearGoal ? 0.8 : 0.3} 
                    />
                  </Box>
                  <Box args={[0.015, 0.012, 0.018]} position={[-0.02, -0.08, 0]}>
                    <meshStandardMaterial 
                      color={isNearGoal ? "#00ff6a" : "#00aa44"} 
                      emissive={isNearGoal ? "#00ff6a" : "#004422"} 
                      emissiveIntensity={isNearGoal ? 0.8 : 0.3} 
                    />
                  </Box>
                  
                  {/* Status light */}
                  <Sphere args={[0.008, 8, 8]} position={[0, 0.015, 0.02]}>
                    <meshStandardMaterial 
                      color={isNearGoal ? "#00ff6a" : "#ffaa00"} 
                      emissive={isNearGoal ? "#00ff6a" : "#ffaa00"} 
                      emissiveIntensity={1} 
                    />
                  </Sphere>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
      
      {/* Goal Target from Trajectory */}
      <group position={[goalPos.x, goalPos.y, goalPos.z]}>
        {/* Target core */}
        <Sphere args={[0.035, 16, 16]}>
          <meshStandardMaterial 
            color={isNearGoal ? "#00ff6a" : "#ff3333"} 
            emissive={isNearGoal ? "#00ff6a" : "#ff0000"} 
            emissiveIntensity={0.6 + pulseRef.current * 0.4}
            transparent 
            opacity={0.9} 
          />
        </Sphere>
        
        {/* Target rings */}
        <mesh rotation={[Math.PI/2, 0, 0]}>
          <ringGeometry args={[0.05, 0.06, 32]} />
          <meshBasicMaterial color={isNearGoal ? "#00ff6a" : "#ff3333"} transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
        <mesh rotation={[0, 0, 0]}>
          <ringGeometry args={[0.05, 0.06, 32]} />
          <meshBasicMaterial color={isNearGoal ? "#00ff6a" : "#ff3333"} transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
        
        {/* Outer pulse ring */}
        <mesh rotation={[Math.PI/2, 0, 0]}>
          <ringGeometry args={[0.08, 0.09, 32]} />
          <meshBasicMaterial color={isNearGoal ? "#00ff6a" : "#ff6666"} transparent opacity={0.2 + pulseRef.current * 0.2} side={THREE.DoubleSide} />
        </mesh>
        
        {/* Distance indicator line to gripper */}
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([0, 0, 0, fkEndPos.x - goalPos.x, fkEndPos.y - goalPos.y, fkEndPos.z - goalPos.z])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={isNearGoal ? "#00ff6a" : "#ff6666"} transparent opacity={0.4} />
        </line>
      </group>
      
      {/* Custom User Targets */}
      {customTargets.map((target, i) => (
        <group key={i} position={[target.x, target.y, target.z]}>
          <Sphere args={[0.03, 12, 12]}>
            <meshStandardMaterial 
              color="#ffaa00" 
              emissive="#ff8800" 
              emissiveIntensity={0.5}
              transparent 
              opacity={0.8} 
            />
          </Sphere>
          <mesh rotation={[Math.PI/2, 0, 0]}>
            <ringGeometry args={[0.04, 0.05, 24]} />
            <meshBasicMaterial color="#ffaa00" transparent opacity={0.5} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}
      
      {/* Trajectory Path */}
      {showPath && positions.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={positions.length}
              array={new Float32Array(positions.flatMap(p => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ff6a" transparent opacity={0.6} />
        </line>
      )}
      
      {/* Visited path points */}
      {showPath && positions.slice(0, currentStep + 1).map((pos, i) => (
        <Sphere key={i} args={[0.006, 6, 6]} position={[pos.x, pos.y, pos.z]}>
          <meshBasicMaterial color="#00ff6a" transparent opacity={0.4 + (i / positions.length) * 0.4} />
        </Sphere>
      ))}
      
      {/* Success indicator when near goal */}
      {isNearGoal && (
        <group position={[goalPos.x, goalPos.y + 0.15, goalPos.z]}>
          <Sphere args={[0.02, 8, 8]}>
            <meshStandardMaterial color="#00ff6a" emissive="#00ff6a" emissiveIntensity={1} />
          </Sphere>
        </group>
      )}
    </group>
  );
}

function Scene3D({ trajectory, currentStep, showPath, customTargets = [], onAddTarget }) {
  return (
    <Canvas 
      shadows 
      camera={{ position: [1.2, 1.0, 1.2], fov: 50 }}
      gl={{ antialias: true }}
      onPointerMissed={(e) => {
        // Optional: Add target on double-click in empty space
        if (e.detail === 2 && onAddTarget) {
          // Could implement raycasting to get 3D position
        }
      }}
    >
      <color attach="background" args={['#050505']} />
      <fog attach="fog" args={['#050505', 4, 10]} />
      
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight 
        position={[5, 8, 5]} 
        intensity={1.2} 
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[-2, 3, -2]} intensity={0.4} color="#00ff6a" />
      <pointLight position={[2, 2, 2]} intensity={0.3} color="#ffffff" />
      <spotLight position={[0, 4, 0]} angle={0.5} penumbra={0.5} intensity={0.5} color="#00ff6a" />
      
      <SawyerArm trajectory={trajectory} currentStep={currentStep} showPath={showPath} customTargets={customTargets} />
      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true}
        minDistance={0.8}
        maxDistance={4}
        target={[0.2, 0.3, 0]}
      />
    </Canvas>
  );
}

// ==================== MAIN APP ====================

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('train');
  
  // Data states
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetDetails, setDatasetDetails] = useState(null);
  const [models, setModels] = useState([]);
  
  // Training states
  const [isTrainingRM, setIsTrainingRM] = useState(false);
  const [isTrainingPolicy, setIsTrainingPolicy] = useState(false);
  const [rmHistory, setRmHistory] = useState({ loss: [], accuracy: [] });
  const [policyHistory, setPolicyHistory] = useState({ reward: [], loss: [] });
  
  // Model Parameters (editable)
  const [rmParams, setRmParams] = useState({
    epochs: 50,
    batch_size: 32,
    learning_rate: 0.0001,
    hidden_dim: 256,
    num_layers: 3,
    dropout: 0.1,
    validation_split: 0.1,
    optimizer: 'adamw',
    loss_type: 'cross_entropy',
    early_stopping_patience: 10
  });
  
  const [policyParams, setPolicyParams] = useState({
    total_steps: 100000,
    steps_per_update: 2048,
    epochs_per_update: 10,
    batch_size: 64,
    learning_rate: 0.0003,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_ratio: 0.2,
    entropy_coef: 0.01,
    value_coef: 0.5,
    max_grad_norm: 0.5,
    hidden_dim: 256,
    num_layers: 2,
    task: 'reach-v3',
    instruction: 'reach the target'
  });
  
  // Visualization states
  const [sampleTrajectories, setSampleTrajectories] = useState([]);
  const [currentTrajIndex, setCurrentTrajIndex] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [rewardDistribution, setRewardDistribution] = useState([]);
  const [customTargets, setCustomTargets] = useState([]);
  const [showTargetPanel, setShowTargetPanel] = useState(false);
  const [newTarget, setNewTarget] = useState({ x: 0.3, y: 0.4, z: 0.2 });
  
  const playIntervalRef = useRef(null);

  // Add custom target
  const addCustomTarget = () => {
    setCustomTargets(prev => [...prev, { ...newTarget, id: Date.now() }]);
  };

  // Remove custom target
  const removeCustomTarget = (id) => {
    setCustomTargets(prev => prev.filter(t => t.id !== id));
  };

  // Clear all custom targets
  const clearCustomTargets = () => {
    setCustomTargets([]);
  };

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      const [datasetsRes, modelsRes] = await Promise.all([
        axios.get(`${API_BASE}/datasets`).catch(() => ({ data: [] })),
        axios.get(`${API_BASE}/models`).catch(() => ({ data: [] })),
      ]);
      setDatasets(datasetsRes.data);
      setModels(modelsRes.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Load sample trajectories when dataset selected
  useEffect(() => {
    if (selectedDataset) {
      loadSampleTrajectories();
    }
  }, [selectedDataset]);

  // Animation playback
  useEffect(() => {
    if (isPlaying && sampleTrajectories.length > 0) {
      const traj = sampleTrajectories[currentTrajIndex]?.traj_a || [];
      playIntervalRef.current = setInterval(() => {
        setCurrentStep(prev => (prev >= traj.length - 1 ? 0 : prev + 1));
      }, 60);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => { if (playIntervalRef.current) clearInterval(playIntervalRef.current); };
  }, [isPlaying, sampleTrajectories, currentTrajIndex]);

  const loadSampleTrajectories = async () => {
    try {
      const res = await axios.get(`${API_BASE}/datasets/${selectedDataset}/samples?n=5`);
      setSampleTrajectories(res.data.samples || []);
      generateRewardDistribution(res.data.samples || []);
    } catch (error) {
      // Generate realistic mock data
      const mockSamples = Array(5).fill(null).map((_, i) => ({
        instruction: ['reach the red target', 'move to goal position', 'touch the marker', 'reach toward target', 'extend to goal'][i],
        traj_a: Array(50).fill(null).map((_, t) => [
          [...Array(36).fill(0).map(() => (Math.random() - 0.5) * 0.3), 0.1 + t * 0.005, 0.15, 0.2 + Math.sin(t * 0.1) * 0.1],
          Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.5)
        ]),
        traj_b: Array(50).fill(null).map((_, t) => [
          [...Array(36).fill(0).map(() => (Math.random() - 0.5) * 0.4), 0.05 + t * 0.002, 0.1 + Math.random() * 0.1, 0.15],
          Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.8)
        ]),
        predicted_preference: Math.random() > 0.3 ? 'A' : 'B'
      }));
      setSampleTrajectories(mockSamples);
      generateRewardDistribution(mockSamples);
    }
  };

  const generateRewardDistribution = (samples) => {
    const dist = samples.flatMap((s, i) => [
      { name: `${i+1}A`, reward: -Math.random() * 50 - 20, type: 'preferred' },
      { name: `${i+1}B`, reward: -Math.random() * 80 - 60, type: 'other' }
    ]);
    setRewardDistribution(dist);
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post(`${API_BASE}/datasets/upload`, formData);
      fetchData();
    } catch (error) {
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const selectDataset = (ds) => {
    setSelectedDataset(ds.name);
    setDatasetDetails(ds);
  };

  const trainRewardModel = async () => {
    if (!selectedDataset) { alert('Select a dataset first'); return; }
    setIsTrainingRM(true);
    setRmHistory({ loss: [], accuracy: [] });
    
    try {
      const res = await axios.post(`${API_BASE}/reward-model/train?dataset_name=${selectedDataset}`, rmParams);
      const jobId = res.data.job_id;
      
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_BASE}/training/${jobId}`);
          const status = statusRes.data;
          
          if (status.train_loss !== null) {
            setRmHistory(prev => ({
              loss: [...prev.loss, { epoch: prev.loss.length + 1, value: status.train_loss }],
              accuracy: [...prev.accuracy, { epoch: prev.accuracy.length + 1, value: (status.train_accuracy || 0) * 100 }]
            }));
          }
          
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
            setIsTrainingRM(false);
            fetchData();
          }
        } catch (e) {}
      }, 1500);
    } catch (error) {
      setIsTrainingRM(false);
      alert(`Training failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const trainPolicy = async () => {
    setIsTrainingPolicy(true);
    setPolicyHistory({ reward: [], loss: [] });
    
    const rewardModel = models.find(m => m.type === 'reward_model');
    
    // Build config from current parameters - ensure task and instruction are included
    const config = {
      task: policyParams.task,
      instruction: policyParams.instruction,
      total_steps: policyParams.total_steps,
      steps_per_update: policyParams.steps_per_update,
      epochs_per_update: policyParams.epochs_per_update,
      batch_size: policyParams.batch_size,
      learning_rate: policyParams.learning_rate,
      gamma: policyParams.gamma,
      gae_lambda: policyParams.gae_lambda,
      clip_ratio: policyParams.clip_ratio,
      entropy_coef: policyParams.entropy_coef,
      value_coef: policyParams.value_coef,
      max_grad_norm: policyParams.max_grad_norm,
      hidden_dim: policyParams.hidden_dim,
      num_layers: policyParams.num_layers,
      reward_model_path: rewardModel?.path || null,
      use_env_reward: !rewardModel,
      // Include custom targets for potential goal conditioning
      custom_goals: customTargets.length > 0 ? customTargets.map(t => [t.x, t.y, t.z]) : null,
    };
    
    console.log('Training policy with config:', config);
    
    try {
      const res = await axios.post(`${API_BASE}/policy/train`, config);
      const jobId = res.data.job_id;
      
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`${API_BASE}/training/${jobId}`);
          const status = statusRes.data;
          
          if (status.cumulative_reward !== null && status.cumulative_reward !== undefined) {
            setPolicyHistory(prev => ({
              reward: [...prev.reward, { step: prev.reward.length + 1, value: status.cumulative_reward }],
              loss: [...prev.loss, { step: prev.loss.length + 1, value: status.policy_loss || 0 }]
            }));
          }
          
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
            setIsTrainingPolicy(false);
            if (status.status === 'failed') {
              alert(`Training failed: ${status.error || 'Unknown error'}`);
            }
            fetchData();
          }
        } catch (e) {
          console.error('Error polling training status:', e);
        }
      }, 2000);
    } catch (error) {
      setIsTrainingPolicy(false);
      alert(`Training failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const currentTrajectory = sampleTrajectories[currentTrajIndex]?.traj_a || [];
  const currentInstruction = sampleTrajectories[currentTrajIndex]?.instruction || policyParams.instruction;

  return (
    <div className="min-h-screen bg-matrix-black grid-bg text-emerald-100">
      {/* Header */}
      <header className="border-b border-neon-green/20 bg-matrix-black/95 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-neon-green/20 border border-neon-green/50 flex items-center justify-center glow-green">
              <Bot className="text-neon-green" size={22} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-neon-green text-glow tracking-wider">RLHF GYM</h1>
              <p className="text-[10px] text-emerald-700 uppercase tracking-widest">MetaWorld Sawyer Training</p>
            </div>
          </div>
          
          <nav className="flex items-center gap-1 bg-matrix-dark rounded-lg p-1 border border-neon-green/10">
            {[
              { id: 'train', icon: Zap, label: 'Train' },
              { id: 'params', icon: Sliders, label: 'Parameters' },
              { id: 'visualize', icon: Eye, label: 'Visualize' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-md font-medium text-sm flex items-center gap-2 transition-all ${
                  activeTab === tab.id
                    ? 'bg-neon-green/20 text-neon-green border border-neon-green/30'
                    : 'text-emerald-600 hover:text-emerald-400'
                }`}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </nav>
          
          <button onClick={fetchData} className="p-2 text-emerald-600 hover:text-neon-green transition">
            <RefreshCw size={18} />
          </button>
        </div>
      </header>

      <main className="max-w-[1920px] mx-auto p-4">
        {/* ==================== TRAIN TAB ==================== */}
        {activeTab === 'train' && (
          <div className="grid grid-cols-12 gap-4 animate-fadeIn">
            {/* Left - Dataset & Quick Config */}
            <div className="col-span-3 space-y-4">
              {/* Dataset */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-2 text-neon-green">
                  <Database size={16} /> DATASET
                </h2>
                
                <label className="block w-full p-3 border border-dashed border-neon-green/30 rounded-lg hover:border-neon-green/60 transition cursor-pointer mb-3 bg-black/30">
                  <div className="text-center">
                    <Upload className="mx-auto mb-1 text-emerald-600" size={20} />
                    <p className="text-xs text-emerald-600">Upload .pkl / .csv</p>
                  </div>
                  <input type="file" accept=".pkl,.pickle,.csv" onChange={handleUpload} className="hidden" />
                </label>
                
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {datasets.map(ds => (
                    <button
                      key={ds.name}
                      onClick={() => selectDataset(ds)}
                      className={`w-full text-left p-2 rounded-lg text-xs transition ${
                        selectedDataset === ds.name
                          ? 'bg-neon-green/20 border border-neon-green/40 text-neon-green'
                          : 'bg-black/30 hover:bg-black/50 border border-transparent text-emerald-400'
                      }`}
                    >
                      <p className="font-medium">{ds.name}</p>
                      <p className="text-emerald-700">{ds.num_pairs?.toLocaleString()} pairs</p>
                    </button>
                  ))}
                </div>
                
                {datasetDetails && (
                  <div className="mt-3 p-2 bg-black/40 rounded-lg text-[10px] grid grid-cols-2 gap-1 text-emerald-600">
                    <span>Instructions: {datasetDetails.num_instructions}</span>
                    <span>Obs: {datasetDetails.obs_dim}D</span>
                    <span>Act: {datasetDetails.act_dim}D</span>
                  </div>
                )}
              </div>

              {/* Quick Actions */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4 space-y-3">
                <h2 className="text-sm font-semibold mb-2 flex items-center gap-2 text-neon-green">
                  <Zap size={16} /> QUICK TRAIN
                </h2>
                
                <button
                  onClick={trainRewardModel}
                  disabled={isTrainingRM || !selectedDataset}
                  className="w-full py-3 bg-gradient-to-r from-neon-green/80 to-emerald-600 text-black font-bold rounded-lg disabled:opacity-40 hover:shadow-lg hover:shadow-neon-green/30 transition flex items-center justify-center gap-2"
                >
                  {isTrainingRM ? (
                    <><div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" /> Training...</>
                  ) : (
                    <><Brain size={16} /> Train Reward Model</>
                  )}
                </button>
                
                <button
                  onClick={trainPolicy}
                  disabled={isTrainingPolicy}
                  className="w-full py-3 bg-gradient-to-r from-emerald-700 to-neon-green-dark text-neon-green font-bold rounded-lg border border-neon-green/30 disabled:opacity-40 hover:shadow-lg hover:shadow-neon-green/20 transition flex items-center justify-center gap-2"
                >
                  {isTrainingPolicy ? (
                    <><div className="w-4 h-4 border-2 border-neon-green/30 border-t-neon-green rounded-full animate-spin" /> Training...</>
                  ) : (
                    <><Bot size={16} /> Train Policy (PPO)</>
                  )}
                </button>
              </div>

              {/* Models */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-2 text-neon-green">
                  <Download size={16} /> MODELS
                </h2>
                
                {models.length === 0 ? (
                  <p className="text-xs text-emerald-700 text-center py-4">No trained models</p>
                ) : (
                  <div className="space-y-2">
                    {models.map((m, i) => (
                      <div key={i} className="flex items-center justify-between p-2 bg-black/30 rounded-lg">
                        <div className="flex items-center gap-2">
                          {m.type === 'reward_model' ? <Brain className="text-neon-green" size={14} /> : <Bot className="text-emerald-500" size={14} />}
                          <span className="text-xs">{m.name}</span>
                        </div>
                        <a href={`${API_BASE}/models/${m.type}/${m.name}/download`} className="text-neon-green hover:text-white">
                          <Download size={14} />
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Center - Training Visualization */}
            <div className="col-span-6 space-y-4">
              {/* Reward Model Charts */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-2 text-neon-green">
                  <Activity size={16} /> REWARD MODEL TRAINING
                </h2>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black/40 rounded-lg p-3">
                    <p className="text-[10px] text-emerald-600 mb-2 uppercase">Loss</p>
                    <div className="h-36">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={rmHistory.loss}>
                          <defs>
                            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#00ff6a" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#00ff6a" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                          <XAxis dataKey="epoch" stroke="#2d4a2d" fontSize={9} />
                          <YAxis stroke="#2d4a2d" fontSize={9} />
                          <Tooltip contentStyle={{ background: '#0a120a', border: '1px solid #00ff6a33', borderRadius: 8, fontSize: 10 }} />
                          <Area type="monotone" dataKey="value" stroke="#00ff6a" fill="url(#lossGrad)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div className="bg-black/40 rounded-lg p-3">
                    <p className="text-[10px] text-emerald-600 mb-2 uppercase">Accuracy %</p>
                    <div className="h-36">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rmHistory.accuracy}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                          <XAxis dataKey="epoch" stroke="#2d4a2d" fontSize={9} />
                          <YAxis stroke="#2d4a2d" fontSize={9} domain={[0, 100]} />
                          <Tooltip contentStyle={{ background: '#0a120a', border: '1px solid #00ff6a33', borderRadius: 8, fontSize: 10 }} />
                          <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              {/* Policy Charts */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-2 text-neon-green">
                  <BarChart3 size={16} /> POLICY TRAINING (PPO)
                </h2>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black/40 rounded-lg p-3">
                    <p className="text-[10px] text-emerald-600 mb-2 uppercase">Episode Reward</p>
                    <div className="h-36">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={policyHistory.reward}>
                          <defs>
                            <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#00cc55" stopOpacity={0.4}/>
                              <stop offset="95%" stopColor="#00cc55" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                          <XAxis dataKey="step" stroke="#2d4a2d" fontSize={9} />
                          <YAxis stroke="#2d4a2d" fontSize={9} />
                          <Tooltip contentStyle={{ background: '#0a120a', border: '1px solid #00ff6a33', borderRadius: 8, fontSize: 10 }} />
                          <Area type="monotone" dataKey="value" stroke="#00cc55" fill="url(#rewardGrad)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div className="bg-black/40 rounded-lg p-3">
                    <p className="text-[10px] text-emerald-600 mb-2 uppercase">Policy Loss</p>
                    <div className="h-36">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={policyHistory.loss}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                          <XAxis dataKey="step" stroke="#2d4a2d" fontSize={9} />
                          <YAxis stroke="#2d4a2d" fontSize={9} />
                          <Tooltip contentStyle={{ background: '#0a120a', border: '1px solid #00ff6a33', borderRadius: 8, fontSize: 10 }} />
                          <Line type="monotone" dataKey="value" stroke="#84cc16" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              {/* Reward Distribution */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h2 className="text-sm font-semibold mb-3 text-neon-green">REWARD DISTRIBUTION</h2>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rewardDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                      <XAxis dataKey="name" stroke="#2d4a2d" fontSize={9} />
                      <YAxis stroke="#2d4a2d" fontSize={9} />
                      <Tooltip contentStyle={{ background: '#0a120a', border: '1px solid #00ff6a33', borderRadius: 8, fontSize: 10 }} />
                      <Bar dataKey="reward" fill="#00ff6a" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Right - 3D Preview */}
            <div className="col-span-3">
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 overflow-hidden h-full flex flex-col">
                {/* Task Badge */}
                <div className="px-3 py-2 bg-gradient-to-r from-neon-green/10 to-transparent border-b border-neon-green/20">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-emerald-600 uppercase">Task:</span>
                    <span className="text-xs font-bold text-neon-green">{policyParams.task}</span>
                  </div>
                  <p className="text-[10px] text-emerald-700 truncate">{currentInstruction}</p>
                </div>
                
                <div className="p-3 border-b border-neon-green/10 flex items-center justify-between">
                  <h2 className="text-sm font-semibold flex items-center gap-2 text-neon-green">
                    <Layers size={14} /> SAWYER ARM
                  </h2>
                  <button
                    onClick={() => setShowPath(!showPath)}
                    className={`text-[10px] px-2 py-1 rounded ${showPath ? 'bg-neon-green/20 text-neon-green' : 'text-emerald-700'}`}
                  >
                    PATH
                  </button>
                </div>
                <div className="flex-1 min-h-[350px]">
                  <Scene3D trajectory={currentTrajectory} currentStep={currentStep} showPath={showPath} customTargets={customTargets} />
                </div>
                <div className="p-3 border-t border-neon-green/10 flex items-center gap-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-2 rounded-lg bg-neon-green/20 text-neon-green hover:bg-neon-green/30"
                  >
                    {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, currentTrajectory.length - 1)}
                    value={currentStep}
                    onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-[10px] text-emerald-600 w-10 text-right">{currentStep}/{currentTrajectory.length || 0}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================== PARAMETERS TAB ==================== */}
        {activeTab === 'params' && (
          <div className="grid grid-cols-2 gap-6 animate-fadeIn max-w-5xl mx-auto">
            {/* Reward Model Parameters */}
            <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold flex items-center gap-2 text-neon-green">
                  <Brain size={20} /> Reward Model Parameters
                </h2>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <ParamInput label="Epochs" value={rmParams.epochs} onChange={v => setRmParams(p => ({...p, epochs: v}))} min={1} max={500} />
                <ParamInput label="Batch Size" value={rmParams.batch_size} onChange={v => setRmParams(p => ({...p, batch_size: v}))} min={1} max={512} />
                <ParamInput label="Learning Rate" value={rmParams.learning_rate} onChange={v => setRmParams(p => ({...p, learning_rate: v}))} step="0.0001" />
                <ParamInput label="Hidden Dim" value={rmParams.hidden_dim} onChange={v => setRmParams(p => ({...p, hidden_dim: v}))} min={32} max={1024} />
                <ParamInput label="Hidden Layers" value={rmParams.num_layers} onChange={v => setRmParams(p => ({...p, num_layers: v}))} min={1} max={10} />
                <ParamInput label="Dropout Rate" value={rmParams.dropout} onChange={v => setRmParams(p => ({...p, dropout: v}))} step="0.01" min={0} max={0.9} />
                <ParamInput label="Validation Split" value={rmParams.validation_split} onChange={v => setRmParams(p => ({...p, validation_split: v}))} step="0.05" min={0.05} max={0.5} />
                <ParamInput label="Optimizer" value={rmParams.optimizer} onChange={v => setRmParams(p => ({...p, optimizer: v}))} options={['adam', 'adamw', 'sgd', 'rmsprop']} />
                <ParamInput label="Loss Type" value={rmParams.loss_type} onChange={v => setRmParams(p => ({...p, loss_type: v}))} options={['cross_entropy', 'hinge', 'bce']} />
                <ParamInput label="Early Stop Patience" value={rmParams.early_stopping_patience} onChange={v => setRmParams(p => ({...p, early_stopping_patience: v}))} min={1} max={50} />
              </div>
            </div>

            {/* Policy Parameters */}
            <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold flex items-center gap-2 text-neon-green">
                  <Bot size={20} /> Policy Parameters (PPO)
                </h2>
              </div>
              
              {/* Task Selection with Description */}
              <div className="mb-4 p-3 bg-black/40 rounded-lg border border-neon-green/10">
                <div className="grid grid-cols-2 gap-4 mb-3">
                  <ParamInput label="Task" value={policyParams.task} onChange={v => setPolicyParams(p => ({...p, task: v}))} options={METAWORLD_TASKS} />
                  <ParamInput label="Total Steps" value={policyParams.total_steps} onChange={v => setPolicyParams(p => ({...p, total_steps: v}))} min={10000} max={10000000} />
                </div>
                <p className="text-xs text-emerald-600 italic">
                  {TASK_DESCRIPTIONS[policyParams.task] || 'Select a task'}
                </p>
              </div>
              
              {/* Instruction Input - Full Width */}
              <div className="mb-4">
                <label className="text-xs text-emerald-600 uppercase tracking-wider block mb-1">Instruction (Natural Language)</label>
                <input
                  type="text"
                  value={policyParams.instruction}
                  onChange={(e) => setPolicyParams(p => ({...p, instruction: e.target.value}))}
                  placeholder="e.g., reach the red target quickly and accurately"
                  className="w-full bg-black/50 border border-neon-green/20 rounded-lg px-3 py-2 text-sm text-emerald-100 focus:border-neon-green focus:outline-none"
                />
                <p className="text-[10px] text-emerald-700 mt-1">This instruction conditions the reward model and policy learning.</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <ParamInput label="Learning Rate" value={policyParams.learning_rate} onChange={v => setPolicyParams(p => ({...p, learning_rate: v}))} step="0.0001" />
                <ParamInput label="Batch Size" value={policyParams.batch_size} onChange={v => setPolicyParams(p => ({...p, batch_size: v}))} min={16} max={512} />
                <ParamInput label="Hidden Dim" value={policyParams.hidden_dim} onChange={v => setPolicyParams(p => ({...p, hidden_dim: v}))} min={32} max={1024} />
                <ParamInput label="Hidden Layers" value={policyParams.num_layers} onChange={v => setPolicyParams(p => ({...p, num_layers: v}))} min={1} max={10} />
                <ParamInput label="Gamma (γ)" value={policyParams.gamma} onChange={v => setPolicyParams(p => ({...p, gamma: v}))} step="0.01" min={0.9} max={0.999} />
                <ParamInput label="GAE Lambda" value={policyParams.gae_lambda} onChange={v => setPolicyParams(p => ({...p, gae_lambda: v}))} step="0.01" min={0.9} max={1} />
                <ParamInput label="Clip Ratio" value={policyParams.clip_ratio} onChange={v => setPolicyParams(p => ({...p, clip_ratio: v}))} step="0.05" min={0.1} max={0.5} />
                <ParamInput label="Entropy Coef" value={policyParams.entropy_coef} onChange={v => setPolicyParams(p => ({...p, entropy_coef: v}))} step="0.001" min={0} max={0.1} />
                <ParamInput label="Value Coef" value={policyParams.value_coef} onChange={v => setPolicyParams(p => ({...p, value_coef: v}))} step="0.1" min={0.1} max={1} />
                <ParamInput label="Max Grad Norm" value={policyParams.max_grad_norm} onChange={v => setPolicyParams({...policyParams, max_grad_norm: v})} step="0.1" min={0.1} max={5} />
              </div>
              
              <div className="mt-4">
                <ParamInput 
                  label="Instruction" 
                  value={policyParams.instruction} 
                  onChange={v => setPolicyParams({...policyParams, instruction: v})} 
                  type="text"
                />
              </div>
            </div>
          </div>
        )}

        {/* ==================== VISUALIZE TAB ==================== */}
        {activeTab === 'visualize' && (
          <div className="grid grid-cols-12 gap-4 animate-fadeIn">
            {/* Main 3D View */}
            <div className="col-span-9">
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 overflow-hidden">
                {/* Task Info Header */}
                <div className="p-3 bg-gradient-to-r from-neon-green/10 to-transparent border-b border-neon-green/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-emerald-600 uppercase">Active Task:</span>
                        <span className="text-sm font-bold text-neon-green">{policyParams.task}</span>
                      </div>
                      <p className="text-xs text-emerald-700 mt-0.5">{TASK_DESCRIPTIONS[policyParams.task]}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-emerald-600 uppercase">Instruction:</div>
                      <p className="text-xs text-emerald-400 max-w-xs truncate">{policyParams.instruction}</p>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 border-b border-neon-green/10 flex items-center justify-between">
                  <h2 className="text-lg font-semibold flex items-center gap-2 text-neon-green">
                    <Eye size={18} /> SAWYER ARM SIMULATION
                  </h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowTargetPanel(!showTargetPanel)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 ${showTargetPanel ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'bg-black/30 text-emerald-600 hover:text-emerald-400'}`}
                    >
                      <Target size={12} /> Targets {customTargets.length > 0 && `(${customTargets.length})`}
                    </button>
                    <button
                      onClick={() => setShowPath(!showPath)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium ${showPath ? 'bg-neon-green/20 text-neon-green border border-neon-green/30' : 'bg-black/30 text-emerald-600'}`}
                    >
                      {showPath ? 'Path ON' : 'Path OFF'}
                    </button>
                  </div>
                </div>
                
                {/* Target Panel (collapsible) */}
                {showTargetPanel && (
                  <div className="p-4 border-b border-neon-green/10 bg-black/40">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-orange-400 flex items-center gap-2">
                        <Target size={14} /> Custom Targets
                      </h3>
                      {customTargets.length > 0 && (
                        <button onClick={clearCustomTargets} className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1">
                          <Trash2 size={12} /> Clear All
                        </button>
                      )}
                    </div>
                    
                    <div className="flex items-end gap-3 mb-3">
                      <div className="flex-1 grid grid-cols-3 gap-2">
                        <div>
                          <label className="text-[10px] text-emerald-600 uppercase">X</label>
                          <input
                            type="number"
                            step="0.05"
                            value={newTarget.x}
                            onChange={(e) => setNewTarget(prev => ({ ...prev, x: parseFloat(e.target.value) }))}
                            className="w-full bg-black/50 border border-neon-green/20 rounded px-2 py-1 text-xs text-emerald-100"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-emerald-600 uppercase">Y (Height)</label>
                          <input
                            type="number"
                            step="0.05"
                            value={newTarget.y}
                            onChange={(e) => setNewTarget(prev => ({ ...prev, y: parseFloat(e.target.value) }))}
                            className="w-full bg-black/50 border border-neon-green/20 rounded px-2 py-1 text-xs text-emerald-100"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-emerald-600 uppercase">Z</label>
                          <input
                            type="number"
                            step="0.05"
                            value={newTarget.z}
                            onChange={(e) => setNewTarget(prev => ({ ...prev, z: parseFloat(e.target.value) }))}
                            className="w-full bg-black/50 border border-neon-green/20 rounded px-2 py-1 text-xs text-emerald-100"
                          />
                        </div>
                      </div>
                      <button
                        onClick={addCustomTarget}
                        className="px-3 py-1.5 bg-orange-500/20 text-orange-400 border border-orange-500/30 rounded-lg text-xs font-medium flex items-center gap-1 hover:bg-orange-500/30"
                      >
                        <Plus size={12} /> Add
                      </button>
                    </div>
                    
                    {customTargets.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {customTargets.map((t, i) => (
                          <div key={t.id} className="flex items-center gap-2 bg-black/30 rounded-lg px-2 py-1 text-xs">
                            <span className="text-orange-400">#{i + 1}</span>
                            <span className="text-emerald-600">({t.x.toFixed(2)}, {t.y.toFixed(2)}, {t.z.toFixed(2)})</span>
                            <button onClick={() => removeCustomTarget(t.id)} className="text-red-400 hover:text-red-300">
                              <X size={12} />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    <p className="text-[10px] text-emerald-700 mt-2">
                      💡 Custom targets (orange) help visualize arm reach. The red target is from trajectory data.
                    </p>
                  </div>
                )}
                
                <div className="h-[500px]">
                  <Scene3D trajectory={currentTrajectory} currentStep={currentStep} showPath={showPath} customTargets={customTargets} />
                </div>
                <div className="p-4 border-t border-neon-green/10 flex items-center gap-4">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-3 rounded-xl bg-neon-green text-black hover:shadow-lg hover:shadow-neon-green/40 transition"
                  >
                    {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, currentTrajectory.length - 1)}
                    value={currentStep}
                    onChange={(e) => { setCurrentStep(parseInt(e.target.value)); setIsPlaying(false); }}
                    className="flex-1"
                  />
                  <span className="text-sm text-emerald-400 font-mono w-24 text-right">
                    Step {currentStep + 1} / {currentTrajectory.length || 1}
                  </span>
                </div>
              </div>
            </div>

            {/* Right sidebar */}
            <div className="col-span-3 space-y-4">
              {/* Trajectory Selector */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h3 className="text-sm font-semibold mb-3 text-neon-green">TRAJECTORY PAIRS</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {sampleTrajectories.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => { setCurrentTrajIndex(i); setCurrentStep(0); setIsPlaying(false); }}
                      className={`w-full text-left p-2 rounded-lg text-xs transition ${
                        currentTrajIndex === i
                          ? 'bg-neon-green/20 border border-neon-green/40'
                          : 'bg-black/30 hover:bg-black/50 border border-transparent'
                      }`}
                    >
                      <p className="font-medium truncate">{s.instruction}</p>
                      <p className="text-emerald-700 mt-1">
                        Preferred: <span className={s.predicted_preference === 'A' ? 'text-neon-green' : 'text-orange-400'}>{s.predicted_preference}</span>
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Position Data */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h3 className="text-sm font-semibold mb-3 text-neon-green">LIVE DATA</h3>
                {currentTrajectory[currentStep] && (
                  <div className="space-y-3 font-mono text-xs">
                    <div>
                      <p className="text-emerald-700 mb-1">End Effector (XYZ)</p>
                      <p className="text-neon-green">
                        [{currentTrajectory[currentStep][0].slice(0, 3).map(v => v.toFixed(4)).join(', ')}]
                      </p>
                    </div>
                    <div>
                      <p className="text-emerald-700 mb-1">Goal Position</p>
                      <p className="text-red-400">
                        [{currentTrajectory[currentStep][0].slice(-3).map(v => v.toFixed(4)).join(', ')}]
                      </p>
                    </div>
                    <div>
                      <p className="text-emerald-700 mb-1">Action</p>
                      <p className="text-lime-400">
                        [{currentTrajectory[currentStep][1].map(v => v.toFixed(4)).join(', ')}]
                      </p>
                    </div>
                    {customTargets.length > 0 && (
                      <div className="pt-2 border-t border-neon-green/10">
                        <p className="text-emerald-700 mb-1">Custom Targets</p>
                        <p className="text-orange-400">{customTargets.length} placed</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 2D Plot */}
              <div className="bg-matrix-card rounded-xl border border-neon-green/10 p-4">
                <h3 className="text-sm font-semibold mb-3 text-neon-green">X-Y PATH</h3>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1a2f1a" />
                      <XAxis type="number" dataKey="x" stroke="#2d4a2d" fontSize={9} domain={[-0.5, 0.5]} />
                      <YAxis type="number" dataKey="y" stroke="#2d4a2d" fontSize={9} domain={[-0.5, 0.5]} />
                      <Scatter
                        data={currentTrajectory.map((step) => ({ x: step[0][0], y: step[0][1] }))}
                        fill="#00ff6a"
                        line={{ stroke: '#00ff6a44', strokeWidth: 1 }}
                      />
                      {currentTrajectory[0] && (
                        <Scatter
                          data={[{ x: currentTrajectory[0][0].slice(-3)[0], y: currentTrajectory[0][0].slice(-3)[1] }]}
                          fill="#ff4444"
                        />
                      )}
                      {/* Custom targets on 2D plot */}
                      <Scatter
                        data={customTargets.map(t => ({ x: t.x / 2.5, y: t.z / 2.5 }))}
                        fill="#ffaa00"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
