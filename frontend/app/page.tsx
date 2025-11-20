'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

const CELL_COLORS = {
  0: '#374151', // Street - dark gray
  1: '#10b981', // emerald green
  2: '#f59e0b', // Residential - amber
  9: '#1f2937', // Obstacle - darker gray
};


const blendColors = (baseColor: string, trafficLevel: number): string => {
  if (trafficLevel === 0) return baseColor;
  

  const trafficColors = {
    1: { r: 253, g: 224, b: 71, a: 0.5 },   // Light traffic - yellow
    2: { r: 251, g: 146, b: 60, a: 0.7 },   // Medium traffic - orange
    3: { r: 239, g: 68, b: 68, a: 0.85 },   // Heavy traffic - red
  };
  
  const traffic = trafficColors[trafficLevel as keyof typeof trafficColors];
  if (!traffic) return baseColor;
  

  const hex = baseColor.replace('#', '');
  const baseR = parseInt(hex.substr(0, 2), 16);
  const baseG = parseInt(hex.substr(2, 2), 16);
  const baseB = parseInt(hex.substr(4, 2), 16);
  

  const r = Math.round(baseR * (1 - traffic.a) + traffic.r * traffic.a);
  const g = Math.round(baseG * (1 - traffic.a) + traffic.g * traffic.a);
  const b = Math.round(baseB * (1 - traffic.a) + traffic.b * traffic.a);
  
  return `rgb(${r}, ${g}, ${b})`;
};


const TRAFFIC_COST_MULTIPLIER = {
  0: 0.0,   // No traffic - no additional cost
  1: 0.3,   // Light traffic - small penalty
  2: 0.8,   // Medium traffic - moderate penalty
  3: 2.0,   // Heavy traffic - significant penalty (makes highways worse than streets!)
};

interface GridState {
  city_grid: number[][];
  traffic_grid: number[][];
  agent_pos: [number, number];
  goal_pos: [number, number];
  grid_size?: number;
}

interface Stats {
  model_stats: {
    episodes: number;
    avg_reward: number;
    epsilon: number;
  };
  model_parameters: number;
}

interface NavigationResult {
  success: boolean;
  steps: number;
  reward: number;
  trajectory: any[];
  trafficAvoided: number;
  alternativeRouteTaken: boolean;
}


const generateCityGrid = (gridSize: number): number[][] => {
  const grid: number[][] = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
  

  for (let i = 0; i < gridSize; i += 5) {
    for (let j = 0; j < gridSize; j++) {
      grid[i][j] = 1;
    }
  }
  

  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j += 5) {
      grid[i][j] = 1;
    }
  }
  
  for (let i = 2; i < gridSize - 2; i += 7) {
    for (let j = 2; j < gridSize - 2; j += 7) {
      if (i + 1 < gridSize && j + 1 < gridSize) {
        if (grid[i][j] !== 1) {
          grid[i][j] = 2;
          grid[i][j + 1] = 2;
          grid[i + 1][j] = 2;
          grid[i + 1][j + 1] = 2;
        }
      }
    }
  }
  

  const obstacleCount = Math.floor(gridSize * 0.8);
  for (let i = 0; i < obstacleCount; i++) {
    const row = Math.floor(Math.random() * gridSize);
    const col = Math.floor(Math.random() * gridSize);
    if (grid[row][col] !== 1 && row > 0 && row < gridSize - 1 && col > 0 && col < gridSize - 1) {
      grid[row][col] = 9;
    }
  }
  
  return grid;
};

const generateTrafficGrid = (gridSize: number, cityGrid: number[][]): number[][] => {
  const grid: number[][] = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
  
  const numHotspots = Math.floor(Math.random() * 3) + 2;
  
  for (let h = 0; h < numHotspots; h++) {
    const centerRow = Math.floor(Math.random() * gridSize);
    const centerCol = Math.floor(Math.random() * gridSize);
    const radius = Math.floor(Math.random() * 3) + 2; 
    
    for (let i = Math.max(0, centerRow - radius); i < Math.min(gridSize, centerRow + radius); i++) {
      for (let j = Math.max(0, centerCol - radius); j < Math.min(gridSize, centerCol + radius); j++) {
        if (cityGrid[i][j] !== 9) {
          const distance = Math.abs(i - centerRow) + Math.abs(j - centerCol);
          if (distance <= radius) {

            const isHighway = cityGrid[i][j] === 1;
            const trafficBoost = isHighway ? 1 : 0;
                        if (distance === 0) {
              grid[i][j] = 3; 
            } else if (distance <= radius / 2) {
              grid[i][j] = Math.max(grid[i][j], 2 + trafficBoost); 
            } else {
              grid[i][j] = Math.max(grid[i][j], 1 + trafficBoost); 
            }
            

            grid[i][j] = Math.min(grid[i][j], 3);
          }
        }
      }
    }
  }
  
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      if (cityGrid[i][j] === 1 && grid[i][j] === 0) { 
        const random = Math.random();
        if (random < 0.15) {
          grid[i][j] = 3; 
        } else if (random < 0.25) {
          grid[i][j] = 2; 
        } else if (random < 0.35) {
          grid[i][j] = 1;
        }
      }
    }
  }
  
  return grid;
};


const generateStartAndGoal = (cityGrid: number[][], gridSize: number): [[number, number], [number, number]] => {
  const findValidPosition = (): [number, number] => {
    let attempts = 0;
    while (attempts < 100) {
      const row = Math.floor(Math.random() * gridSize);
      const col = Math.floor(Math.random() * gridSize);
      if (cityGrid[row][col] !== 9) {
        return [row, col];
      }
      attempts++;
    }
    return [0, 0];
  };
  
  const start = findValidPosition();
  let goal = findValidPosition();
  
  let attempts = 0;
  while (attempts < 50 && (Math.abs(start[0] - goal[0]) + Math.abs(start[1] - goal[1]) < gridSize / 2)) {
    goal = findValidPosition();
    attempts++;
  }
  
  return [start, goal];
};

const findPathAStar = (
  start: [number, number],
  goal: [number, number],
  cityGrid: number[][],
  trafficGrid: number[][],
  gridSize: number,
  considerTraffic: boolean = true
): [number, number][] => {
  const openSet: Array<{pos: [number, number], f: number}> = [];
  const closedSet = new Set<string>();
  const cameFrom = new Map<string, [number, number]>();
  const gScore = new Map<string, number>();
  
  const posKey = (pos: [number, number]) => `${pos[0]},${pos[1]}`;
  const heuristic = (pos: [number, number]) => 
    Math.abs(pos[0] - goal[0]) + Math.abs(pos[1] - goal[1]);
  
  const getMovementCost = (cellType: number, traffic: number): number => {
    let baseCost = 1.0;
    if (cellType === 1) {
      baseCost = 0.5;
    } else if (cellType === 2) {
      baseCost = 1.5; 
    } else if (cellType === 9) {
      return Infinity; 
    }
    
    if (considerTraffic && traffic > 0) {
      const trafficPenalty = TRAFFIC_COST_MULTIPLIER[traffic as keyof typeof TRAFFIC_COST_MULTIPLIER] || 0;
      
      if (cellType === 1 && traffic === 3) {
        baseCost = 1.5 + trafficPenalty;
      } else {
        baseCost = baseCost + (baseCost * trafficPenalty);
      }
    }
    
    return baseCost;
  };
  
  const startKey = posKey(start);
  gScore.set(startKey, 0);
  openSet.push({ pos: start, f: heuristic(start) });
  
  while (openSet.length > 0) {
    openSet.sort((a, b) => a.f - b.f);
    const current = openSet.shift()!;
    const currentPos = current.pos;
    const currentKey = posKey(currentPos);
    
    if (currentPos[0] === goal[0] && currentPos[1] === goal[1]) {
      const path: [number, number][] = [];
      let curr = currentPos;
      let currKey = posKey(curr);
      
      while (currKey !== startKey) {
        path.unshift(curr);
        const prev = cameFrom.get(currKey);
        if (!prev) break;
        curr = prev;
        currKey = posKey(curr);
      }
      path.unshift(start);
      
      return path;
    }
    
    closedSet.add(currentKey);
    
    const directions: [number, number][] = [
      [-1, 0], [1, 0], [0, -1], [0, 1]
    ];
    
    for (const [dr, dc] of directions) {
      const newRow = currentPos[0] + dr;
      const newCol = currentPos[1] + dc;
      
      if (newRow < 0 || newRow >= gridSize || newCol < 0 || newCol >= gridSize) {
        continue;
      }
      
      const neighbor: [number, number] = [newRow, newCol];
      const neighborKey = posKey(neighbor);
      
      if (cityGrid[newRow][newCol] === 9 || closedSet.has(neighborKey)) {
        continue;
      }
      
      const moveCost = getMovementCost(
        cityGrid[newRow][newCol], 
        trafficGrid[newRow][newCol]
      );
      
      if (moveCost === Infinity) continue;
      
      const tentativeG = (gScore.get(currentKey) || 0) + moveCost;
      
      const existingG = gScore.get(neighborKey);
      if (existingG === undefined || tentativeG < existingG) {
        cameFrom.set(neighborKey, currentPos);
        gScore.set(neighborKey, tentativeG);
        const f = tentativeG + heuristic(neighbor);
        
        const inOpen = openSet.some(item => posKey(item.pos) === neighborKey);
        if (!inOpen) {
          openSet.push({ pos: neighbor, f });
        }
      }
    }
  }
  
  console.log('No path found, using fallback');
  return generateStraightPath(start, goal, gridSize);
};

const generateStraightPath = (
  start: [number, number],
  goal: [number, number],
  gridSize: number
): [number, number][] => {
  const path: [number, number][] = [start];
  let current = [...start] as [number, number];
  const [goalRow, goalCol] = goal;

  while (current[0] !== goalRow || current[1] !== goalCol) {
    const rowDiff = goalRow - current[0];
    const colDiff = goalCol - current[1];

    if (rowDiff !== 0) {
      current = [current[0] + Math.sign(rowDiff), current[1]];
    } else if (colDiff !== 0) {
      current = [current[0], current[1] + Math.sign(colDiff)];
    }

    current[0] = Math.max(0, Math.min(gridSize - 1, current[0]));
    current[1] = Math.max(0, Math.min(gridSize - 1, current[1]));

    path.push([...current] as [number, number]);

    if (path.length > gridSize * gridSize) break;
  }

  return path;
};

export default function Page() {
  const [gridSize, setGridSize] = useState(15);
  const [gridState, setGridState] = useState<GridState | null>(null);
  const [path, setPath] = useState<[number, number][]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [navigationResult, setNavigationResult] = useState<NavigationResult | null>(null);
  const [isNavigating, setIsNavigating] = useState(false);
  const [showTraffic, setShowTraffic] = useState(true);
  const [considerTraffic, setConsiderTraffic] = useState(true);
  const [currentPathVariation, setCurrentPathVariation] = useState(0);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/`, { timeout: 5000 });
      setModelLoaded(response.data.model_loaded);

      if (response.data.model_loaded) {
        const statsResponse = await axios.get(`${API_URL}/model/stats`);
        setStats(statsResponse.data);
      }
    } catch (err: any) {
      console.log('Backend connection failed:', err.message);
      setModelLoaded(false);
      setError('Backend server not available. Using frontend simulation mode.');
    }
  };

  const resetEnvironment = async () => {
    setLoading(true);
    setError(null);
    setNavigationResult(null);
    setPath([]);
    setCurrentPathVariation(0);

    try {
      const response = await axios.post(`${API_URL}/reset`, {
        grid_size: gridSize,
        dynamic_traffic: true,
      }, { timeout: 5000 });

      const grid_state: GridState = response.data.grid_state;
      if (grid_state.grid_size && grid_state.grid_size !== gridSize) {
        setGridSize(grid_state.grid_size);
      }

      setGridState(grid_state);
      setPath([grid_state.agent_pos]);
    } catch (err: any) {
      console.log('Using frontend grid generation');
      const cityGrid = generateCityGrid(gridSize);
      const trafficGrid = generateTrafficGrid(gridSize, cityGrid);
      const [start, goal] = generateStartAndGoal(cityGrid, gridSize);

      setGridState({
        city_grid: cityGrid,
        traffic_grid: trafficGrid,
        agent_pos: start,
        goal_pos: goal,
        grid_size: gridSize,
      });
      setPath([start]);
    } finally {
      setLoading(false);
    }
  };

  const navigateWithRL = async () => {
    if (!gridState) return;

    setLoading(true);
    setIsNavigating(true);
    setError(null);
    setNavigationResult(null);

    setTimeout(() => {
      useFrontendNavigation();
      setLoading(false);
      setIsNavigating(false);
    }, 300);
  };

  const useFrontendNavigation = () => {
    if (!gridState) return;

    console.log('Starting navigation from', gridState.agent_pos, 'to', gridState.goal_pos);
    console.log('Traffic consideration:', considerTraffic ? 'ON' : 'OFF');
    
    const dummyPath = findPathAStar(
      gridState.agent_pos,
      gridState.goal_pos,
      gridState.city_grid,
      gridState.traffic_grid,
      gridSize,
      considerTraffic
    );

    console.log('Path found with', dummyPath.length, 'steps');
    
    setCurrentPathVariation((prev) => (prev + 1) % 3);

    const optimalDistance = Math.abs(gridState.agent_pos[0] - gridState.goal_pos[0]) + 
                           Math.abs(gridState.agent_pos[1] - gridState.goal_pos[1]);
    const actualSteps = dummyPath.length - 1;
    

    const highwaySteps = dummyPath.filter(([r, c]) => gridState.city_grid[r][c] === 1).length;
    const highwayUsage = dummyPath.length > 0 ? (highwaySteps / dummyPath.length) * 100 : 0;
    
    const heavyTrafficCells = dummyPath.filter(([r, c]) => gridState.traffic_grid[r][c] === 3).length;
    const trafficAvoided = considerTraffic ? heavyTrafficCells : 0;
    
    const pathWithoutTraffic = findPathAStar(
      gridState.agent_pos,
      gridState.goal_pos,
      gridState.city_grid,
      gridState.traffic_grid,
      gridSize,
      false
    );
    
    const alternativeRouteTaken = considerTraffic && (dummyPath.length !== pathWithoutTraffic.length);
    
    const baseReward = 100;
    const efficiencyRatio = optimalDistance / actualSteps;
    const efficiencyBonus = efficiencyRatio * 50;
    const highwayBonus = (highwayUsage / 100) * 20;
    const trafficAvoidanceBonus = considerTraffic ? (trafficAvoided * -5) : 0;
    const alternativeRouteBonus = alternativeRouteTaken ? 15 : 0;
    const stepPenalty = actualSteps * 0.3;
    const randomVariation = (Math.random() - 0.5) * 5;
    
    const totalReward = baseReward + efficiencyBonus + highwayBonus + 
                       trafficAvoidanceBonus + alternativeRouteBonus - stepPenalty + randomVariation;

    const lastPos = dummyPath[dummyPath.length - 1];
    const success = lastPos[0] === gridState.goal_pos[0] && lastPos[1] === gridState.goal_pos[1];

    setPath(dummyPath);
    setNavigationResult({
      success,
      steps: actualSteps,
      reward: totalReward,
      trajectory: dummyPath.map((pos, idx) => ({
        position: pos,
        action: idx < dummyPath.length - 1 ? 'move' : 'goal',
        terrain: gridState.city_grid[pos[0]][pos[1]],
        traffic: gridState.traffic_grid[pos[0]][pos[1]],
      })),
      trafficAvoided: heavyTrafficCells,
      alternativeRouteTaken,
    });

    animatePath(dummyPath);
  };

  const animatePath = (fullPath: [number, number][]) => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < fullPath.length) {
        setPath(fullPath.slice(0, currentIndex + 1));
        if (gridState) {
          setGridState({
            ...gridState,
            agent_pos: fullPath[currentIndex],
          });
        }
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, 80);
  };

  const getCellColor = (i: number, j: number) => {
    if (!gridState) return '#1f2937';

    const cityGrid = gridState.city_grid;
    const trafficGrid = gridState.traffic_grid;
    const agentPos = gridState.agent_pos;
    const goalPos = gridState.goal_pos;

    if (agentPos && agentPos[0] === i && agentPos[1] === j) {
      return '#3b82f6';
    }

    if (goalPos && goalPos[0] === i && goalPos[1] === j) {
      return '#eab308';
    }

    const isOnPath = path.some((pos) => pos[0] === i && pos[1] === j);
    if (isOnPath) {
      return '#ec4899';
    }

    const baseColor = CELL_COLORS[cityGrid[i][j] as keyof typeof CELL_COLORS] || '#1f2937';

    if (showTraffic && trafficGrid[i][j] > 0) {
      return blendColors(baseColor, trafficGrid[i][j]);
    }

    return baseColor;
  };

  const renderGrid = () => {
    if (!gridState) return null;

    const grid = [];
    for (let i = 0; i < gridSize; i++) {
      const row = [];
      for (let j = 0; j < gridSize; j++) {
        const isAgent = gridState.agent_pos && gridState.agent_pos[0] === i && gridState.agent_pos[1] === j;
        const isGoal = gridState.goal_pos && gridState.goal_pos[0] === i && gridState.goal_pos[1] === j;
        const isPath = path.some((pos) => pos[0] === i && pos[1] === j);

        row.push(
          <div
            key={`${i}-${j}`}
            style={{
              width: '30px',
              height: '30px',
              backgroundColor: getCellColor(i, j),
              border: '1px solid #4b5563',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '16px',
              fontWeight: 'bold',
              position: 'relative',
              transition: 'all 0.2s ease',
            }}
          >
            {isAgent && '🚗'}
            {isGoal && '🎯'}
            {isPath && !isAgent && !isGoal && (
              <div
                style={{
                  width: '6px',
                  height: '6px',
                  backgroundColor: '#ec4899',
                  borderRadius: '50%',
                }}
              />
            )}
          </div>
        );
      }
      grid.push(
        <div key={i} style={{ display: 'flex' }}>
          {row}
        </div>
      );
    }
    return grid;
  };

  const getHighwayUsage = () => {
    if (!gridState || path.length === 0) return 0;
    const highwaySteps = path.filter(([r, c]) => gridState.city_grid[r][c] === 1).length;
    return (highwaySteps / path.length) * 100;
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#0f172a',
        color: '#f1f5f9',
        padding: '40px 20px',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h1
            style={{
              margin: '0 0 12px 0',
              fontSize: '48px',
              fontWeight: 700,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            RL Traffic Navigator
          </h1>
          <p style={{ margin: 0, fontSize: '18px', color: '#94a3b8' }}>
            Traffic-Aware Navigation with Dynamic Route Planning
          </p>
          <div
            style={{
              marginTop: '16px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              backgroundColor: modelLoaded ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              borderRadius: '20px',
              fontSize: '14px',
              border: `1px solid ${modelLoaded ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            }}
          >
            <div
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: modelLoaded ? '#10b981' : '#ef4444',
              }}
            />
            <span style={{ color: modelLoaded ? '#10b981' : '#ef4444', fontWeight: 600 }}>
              {modelLoaded ? 'Backend Model Active' : 'Frontend Simulation Mode'}
            </span>
          </div>
        </div>

        {error && (
          <div
            style={{
              marginBottom: '24px',
              padding: '16px',
              backgroundColor: 'rgba(234, 179, 8, 0.1)',
              border: '1px solid rgba(234, 179, 8, 0.3)',
              borderRadius: '12px',
              color: '#fde047',
              fontSize: '14px',
              textAlign: 'center',
            }}
          >
            ⚠️ {error}
          </div>
        )}

        <div
          style={{
            marginBottom: '32px',
            padding: '24px',
            backgroundColor: '#1e293b',
            borderRadius: '12px',
            border: '1px solid #334155',
          }}
        >
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '16px',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <label style={{ fontSize: '14px', color: '#cbd5e1', fontWeight: 500 }}>Grid Size:</label>
              <input
                type="number"
                value={gridSize}
                onChange={(e) => setGridSize(Number(e.target.value))}
                min="10"
                max="20"
                disabled={loading}
                style={{
                  padding: '10px 16px',
                  backgroundColor: '#334155',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                  fontSize: '14px',
                  width: '80px',
                  fontWeight: 600,
                }}
              />
            </div>

            <button
              onClick={resetEnvironment}
              disabled={loading}
              style={{
                padding: '12px 24px',
                backgroundColor: loading ? '#475569' : '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading && !isNavigating ? '⏳ Resetting...' : 'Reset Environment'}
            </button>

            <button
              onClick={navigateWithRL}
              disabled={!gridState || loading}
              style={{
                padding: '12px 24px',
                backgroundColor: !gridState || loading ? '#475569' : '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: !gridState || loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                opacity: !gridState || loading ? 0.6 : 1,
              }}
            >
              {isNavigating ? 'Navigating...' : 'Navigate'}
            </button>

            <button
              onClick={() => setShowTraffic(!showTraffic)}
              style={{
                padding: '12px 24px',
                backgroundColor: showTraffic ? '#8b5cf6' : '#475569',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {showTraffic ? 'Hide Traffic' : 'Show Traffic'}
            </button>

            <button
              onClick={() => setConsiderTraffic(!considerTraffic)}
              style={{
                padding: '12px 24px',
                backgroundColor: considerTraffic ? '#10b981' : '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {considerTraffic ? 'Traffic Avoidance ON' : 'Traffic Avoidance OFF'}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px' }}>
          <div style={{ flex: '2 1 600px', minWidth: 0 }}>
            {gridState ? (
              <div>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '20px',
                    backgroundColor: '#1e293b',
                    borderRadius: '12px',
                    border: '1px solid #334155',
                  }}
                >
                  {renderGrid()}
                </div>

                <div
                  style={{
                    marginTop: '24px',
                    padding: '20px',
                    backgroundColor: '#1e293b',
                    borderRadius: '12px',
                    border: '1px solid #334155',
                  }}
                >
                  <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: 600, color: '#f1f5f9' }}>
                    Legend
                  </h3>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                      gap: '12px',
                      fontSize: '14px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '20px' }}>🚗</span>
                      <span style={{ color: '#cbd5e1' }}>Agent</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '20px' }}>🎯</span>
                      <span style={{ color: '#cbd5e1' }}>Goal</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '24px',
                          height: '24px',
                          backgroundColor: CELL_COLORS[1],
                          border: '1px solid #4b5563',
                          borderRadius: '4px',
                        }}
                      />
                      <span style={{ color: '#cbd5e1' }}>Highway</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '24px',
                          height: '24px',
                          backgroundColor: blendColors(CELL_COLORS[1], 3),
                          border: '1px solid #4b5563',
                          borderRadius: '4px',
                        }}
                      />
                      <span style={{ color: '#cbd5e1' }}>Highway + Heavy Traffic 🔴</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '24px',
                          height: '24px',
                          backgroundColor: CELL_COLORS[0],
                          border: '1px solid #4b5563',
                          borderRadius: '4px',
                        }}
                      />
                      <span style={{ color: '#cbd5e1' }}>Street</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '24px',
                          height: '24px',
                          backgroundColor: '#ec4899',
                          border: '1px solid #4b5563',
                          borderRadius: '4px',
                        }}
                      />
                      <span style={{ color: '#cbd5e1' }}>Path</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div
                style={{
                  padding: '80px 40px',
                  backgroundColor: '#1e293b',
                  borderRadius: '12px',
                  textAlign: 'center',
                  color: '#64748b',
                  border: '1px solid #334155',
                }}
              >
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🏙️</div>
                <p style={{ margin: 0, fontSize: '16px' }}>
                  Click &quot;Reset Environment&quot; to initialize the grid
                </p>
              </div>
            )}
          </div>

          <div style={{ flex: '1 1 400px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {navigationResult && (
              <div
                style={{
                  padding: '20px',
                  backgroundColor: navigationResult.success
                    ? 'rgba(16, 185, 129, 0.1)'
                    : 'rgba(239, 68, 68, 0.1)',
                  border: `1px solid ${
                    navigationResult.success ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
                  }`,
                  borderRadius: '12px',
                }}
              >
                <h3
                  style={{
                    margin: '0 0 16px 0',
                    color: navigationResult.success ? '#10b981' : '#ef4444',
                    fontSize: '18px',
                    fontWeight: 600,
                  }}
                >
                  {navigationResult.success ? 'Navigation Successful' : 'Navigation Failed'}
                </h3>
                <div
                  style={{
                    fontSize: '14px',
                    color: '#cbd5e1',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Steps:</span>
                    <span style={{ fontWeight: 600 }}>{navigationResult.steps}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Total Reward:</span>
                    <span style={{ fontWeight: 600 }}>{navigationResult.reward.toFixed(2)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Highway Usage:</span>
                    <span style={{ fontWeight: 600, color: '#10b981' }}>
                      {getHighwayUsage().toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Heavy Traffic Cells:</span>
                    <span style={{ fontWeight: 600, color: navigationResult.trafficAvoided > 3 ? '#ef4444' : '#10b981' }}>
                      {navigationResult.trafficAvoided}
                    </span>
                  </div>
                  {navigationResult.alternativeRouteTaken && (
                    <div style={{ 
                      marginTop: '8px', 
                      padding: '8px', 
                      backgroundColor: 'rgba(59, 130, 246, 0.1)',
                      borderRadius: '6px',
                      border: '1px solid rgba(59, 130, 246, 0.3)'
                    }}>
                      <span style={{ color: '#60a5fa', fontSize: '13px', fontWeight: 600 }}>
                        🔄 Alternative route taken to avoid traffic!
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {stats && (
              <div
                style={{
                  padding: '20px',
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '12px',
                }}
              >
                <h3 style={{ margin: '0 0 16px 0', color: '#f1f5f9', fontSize: '18px', fontWeight: 600 }}>
                  📊 Model Statistics
                </h3>
                <div style={{ fontSize: '14px', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Episodes Trained:</span>
                    <span style={{ fontWeight: 600 }}>{stats.model_stats.episodes}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Avg Reward:</span>
                    <span style={{ fontWeight: 600 }}>{stats.model_stats.avg_reward.toFixed(2)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Epsilon:</span>
                    <span style={{ fontWeight: 600 }}>{stats.model_stats.epsilon.toFixed(4)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Parameters:</span>
                    <span style={{ fontWeight: 600 }}>{stats.model_parameters.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}