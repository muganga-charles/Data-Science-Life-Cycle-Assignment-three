'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

const CELL_COLORS = {
  0: '#374151',  // Street - dark gray
  1: '#10b981',  // Highway - emerald green
  2: '#f59e0b',  // Residential - amber
  9: '#1f2937',  // Obstacle - darker gray
};

const TRAFFIC_COLORS = {
  0: 'rgba(0, 0, 0, 0)',
  1: 'rgba(253, 224, 71, 0.4)',
  2: 'rgba(251, 146, 60, 0.6)',
  3: 'rgba(239, 68, 68, 0.8)',
  9: 'rgba(0, 0, 0, 0.9)',
};

interface GridState {
  city_grid: number[][];
  traffic_grid: number[][];
  agent_pos: [number, number];
  goal_pos: [number, number];
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
}

export default function Page() {
  // State
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

  // Check if model is loaded on mount
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
      console.log('[v0] Backend connection failed:', err.message);
      setModelLoaded(false);
      setError('Backend server is not available. The app will work in demo mode once the server is running on port 8000.');
    }
  };

  const resetEnvironment = async () => {
    setLoading(true);
    setError(null);
    setNavigationResult(null);
    setPath([]);
    
    try {
      const response = await axios.post(`${API_URL}/reset`, {
        grid_size: gridSize,
        dynamic_traffic: true
      });
      
      setGridState(response.data.grid_state);
      setPath([response.data.grid_state.agent_pos]);
    } catch (err: any) {
      setError('Failed to reset environment: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const navigateWithRL = async () => {
    setLoading(true);
    setIsNavigating(true);
    setError(null);
    setNavigationResult(null);
    
    try {
      const response = await axios.post(`${API_URL}/navigate`, {
        grid_size: gridSize,
        dynamic_traffic: true
      });
      
      setGridState(response.data.grid_state);
      setPath(response.data.path);
      setNavigationResult({
        success: response.data.success,
        steps: response.data.steps,
        reward: response.data.total_reward,
        trajectory: response.data.trajectory
      });
      
      // Animate the path
      animatePath(response.data.path);
    } catch (err: any) {
      setError('Navigation failed: ' + err.message);
    } finally {
      setLoading(false);
      setIsNavigating(false);
    }
  };

  const animatePath = (fullPath: [number, number][]) => {
    // Optional: animate the path step by step
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < fullPath.length) {
        setPath(fullPath.slice(0, currentIndex + 1));
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, 100);
  };

  const getCellColor = (i: number, j: number) => {
    if (!gridState) return '#1f2937';
    
    const cityGrid = gridState.city_grid;
    const trafficGrid = gridState.traffic_grid;
    const agentPos = gridState.agent_pos;
    const goalPos = gridState.goal_pos;
    
    if (agentPos && agentPos[0] === i && agentPos[1] === j) {
      return '#3b82f6'; // Blue for agent
    }
    
    if (goalPos && goalPos[0] === i && goalPos[1] === j) {
      return '#eab308'; // Yellow for goal
    }
    
    const isOnPath = path.some(pos => pos[0] === i && pos[1] === j);
    if (isOnPath) {
      return '#ec4899'; // Pink for path
    }
    
    const baseColor = CELL_COLORS[cityGrid[i][j] as keyof typeof CELL_COLORS] || '#1f2937';
    
    if (showTraffic && trafficGrid[i][j] > 0) {
      return TRAFFIC_COLORS[trafficGrid[i][j] as keyof typeof TRAFFIC_COLORS] || baseColor;
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
        const isPath = path.some(pos => pos[0] === i && pos[1] === j);
        
        row.push(
          <div
            key={`${i}-${j}`}
            className="grid-cell"
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
              transition: 'all 0.2s ease'
            }}
          >
            {isAgent && '🚗'}
            {isGoal && '🎯'}
            {isPath && !isAgent && !isGoal && <div style={{
              width: '8px',
              height: '8px',
              backgroundColor: '#ec4899',
              borderRadius: '50%',
              boxShadow: '0 0 4px rgba(236, 72, 153, 0.6)'
            }} />}
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

  return (
    <div style={{ 
      minHeight: '100vh',
      backgroundColor: '#0f172a',
      color: '#e2e8f0',
      padding: '32px 20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
        <header style={{ marginBottom: '32px' }}>
          <h1 style={{ 
            fontSize: '32px',
            fontWeight: '700',
            color: '#f1f5f9',
            margin: '0 0 12px 0',
            letterSpacing: '-0.025em'
          }}>
            Smart City RL Navigation System
          </h1>
          <p style={{ 
            color: '#94a3b8',
            fontSize: '16px',
            margin: 0,
            lineHeight: 1.6
          }}>
            Deep Q-Network (DQN) autonomous pathfinding for sustainable urban infrastructure
          </p>
        </header>

        <div style={{
          backgroundColor: modelLoaded ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          border: `1px solid ${modelLoaded ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
          color: modelLoaded ? '#10b981' : '#ef4444',
          padding: '16px',
          borderRadius: '8px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{ 
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: modelLoaded ? '#10b981' : '#ef4444',
            boxShadow: `0 0 8px ${modelLoaded ? '#10b981' : '#ef4444'}`
          }} />
          <span style={{ fontWeight: 600 }}>
            {modelLoaded ? 'Model Loaded - Ready for Navigation' : 'No Model Loaded - Train using train_dqn.py'}
          </span>
        </div>

        <div style={{
          backgroundColor: '#1e293b',
          border: '1px solid #334155',
          borderRadius: '12px',
          padding: '24px',
          marginBottom: '24px'
        }}>
          <div style={{
            display: 'flex',
            gap: '20px',
            flexWrap: 'wrap',
            alignItems: 'center'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ fontSize: '13px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Grid Size
              </label>
              <select
                value={gridSize}
                onChange={(e) => setGridSize(parseInt(e.target.value))}
                style={{
                  padding: '10px 16px',
                  borderRadius: '6px',
                  border: '1px solid #475569',
                  fontSize: '14px',
                  backgroundColor: '#0f172a',
                  color: '#e2e8f0',
                  cursor: 'pointer',
                  fontWeight: 500
                }}
                disabled={loading || isNavigating}
              >
                <option value={10}>10 × 10</option>
                <option value={15}>15 × 15</option>
                <option value={20}>20 × 20</option>
              </select>
            </div>

            <label style={{ 
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              cursor: 'pointer',
              padding: '10px 16px',
              backgroundColor: showTraffic ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
              border: `1px solid ${showTraffic ? 'rgba(59, 130, 246, 0.3)' : '#475569'}`,
              borderRadius: '6px',
              transition: 'all 0.2s'
            }}>
              <input
                type="checkbox"
                checked={showTraffic}
                onChange={(e) => setShowTraffic(e.target.checked)}
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
              <span style={{ fontWeight: 600, fontSize: '14px' }}>Show Traffic Layer</span>
            </label>

            <div style={{ flex: 1 }} />

            <button
              onClick={resetEnvironment}
              disabled={loading || isNavigating}
              style={{
                padding: '12px 24px',
                backgroundColor: '#475569',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: 600,
                transition: 'all 0.2s',
                opacity: loading ? 0.5 : 1
              }}
            >
              Reset Environment
            </button>

            <button
              onClick={navigateWithRL}
              disabled={!modelLoaded || loading || isNavigating || !gridState}
              style={{
                padding: '12px 24px',
                backgroundColor: modelLoaded && gridState ? '#3b82f6' : '#334155',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: modelLoaded && gridState && !loading ? 'pointer' : 'not-allowed',
                fontSize: '14px',
                fontWeight: 600,
                transition: 'all 0.2s',
                opacity: !modelLoaded || !gridState || loading ? 0.5 : 1
              }}
            >
              Navigate with RL Agent
            </button>
          </div>
        </div>

        {error && (
          <div style={{
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            padding: '16px',
            borderRadius: '8px',
            marginBottom: '24px',
            fontSize: '14px',
            lineHeight: 1.6
          }}>
            <strong style={{ display: 'block', marginBottom: '4px', color: '#ef4444' }}>Connection Error</strong>
            {error}
          </div>
        )}

        {loading && (
          <div style={{
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            color: '#60a5fa',
            padding: '16px',
            borderRadius: '8px',
            marginBottom: '24px',
            textAlign: 'center',
            fontWeight: 600
          }}>
            {isNavigating ? '🤖 Agent is navigating...' : 'Loading...'}
          </div>
        )}

        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {/* Grid Visualization */}
          <div style={{ flex: '1 1 600px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: 600, color: '#f1f5f9', marginBottom: '16px' }}>
              City Grid Visualization
            </h2>
            {gridState ? (
              <div>
                <div style={{
                  display: 'inline-block',
                  padding: '20px',
                  backgroundColor: '#1e293b',
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
                  border: '1px solid #334155'
                }}>
                  {renderGrid()}
                </div>

                <div style={{ 
                  marginTop: '24px',
                  padding: '20px',
                  backgroundColor: '#1e293b',
                  borderRadius: '12px',
                  border: '1px solid #334155'
                }}>
                  <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: 600, color: '#f1f5f9' }}>
                    Legend
                  </h3>
                  <div style={{ 
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                    gap: '12px',
                    fontSize: '14px'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '20px' }}>🚗</span>
                      <span style={{ color: '#cbd5e1' }}>Agent</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '20px' }}>🎯</span>
                      <span style={{ color: '#cbd5e1' }}>Goal</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ 
                        width: '24px',
                        height: '24px',
                        backgroundColor: CELL_COLORS[0],
                        border: '1px solid #4b5563',
                        borderRadius: '4px'
                      }} />
                      <span style={{ color: '#cbd5e1' }}>Street</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ 
                        width: '24px',
                        height: '24px',
                        backgroundColor: CELL_COLORS[1],
                        border: '1px solid #4b5563',
                        borderRadius: '4px'
                      }} />
                      <span style={{ color: '#cbd5e1' }}>Highway</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ 
                        width: '24px',
                        height: '24px',
                        backgroundColor: CELL_COLORS[2],
                        border: '1px solid #4b5563',
                        borderRadius: '4px'
                      }} />
                      <span style={{ color: '#cbd5e1' }}>Residential</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ 
                        width: '24px',
                        height: '24px',
                        backgroundColor: CELL_COLORS[9],
                        border: '1px solid #4b5563',
                        borderRadius: '4px'
                      }} />
                      <span style={{ color: '#cbd5e1' }}>Obstacle</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ 
                        width: '24px',
                        height: '24px',
                        backgroundColor: '#ec4899',
                        border: '1px solid #4b5563',
                        borderRadius: '4px'
                      }} />
                      <span style={{ color: '#cbd5e1' }}>Path</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                padding: '80px 40px',
                backgroundColor: '#1e293b',
                borderRadius: '12px',
                textAlign: 'center',
                color: '#64748b',
                border: '1px solid #334155'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🏙️</div>
                <p style={{ margin: 0, fontSize: '16px' }}>Click &quot;Reset Environment&quot; to initialize the grid</p>
              </div>
            )}
          </div>

          <div style={{ flex: '1 1 400px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Navigation Results */}
            {navigationResult && (
              <div style={{
                padding: '20px',
                backgroundColor: navigationResult.success ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${navigationResult.success ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                borderRadius: '12px'
              }}>
                <h3 style={{ 
                  margin: '0 0 16px 0',
                  color: navigationResult.success ? '#10b981' : '#ef4444',
                  fontSize: '18px',
                  fontWeight: 600
                }}>
                  {navigationResult.success ? 'Navigation Successful' : 'Navigation Failed'}
                </h3>
                <div style={{ 
                  fontSize: '14px',
                  color: '#cbd5e1',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Steps:</span>
                    <span style={{ fontWeight: 600 }}>{navigationResult.steps}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Total Reward:</span>
                    <span style={{ fontWeight: 600 }}>{navigationResult.reward.toFixed(2)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Path Length:</span>
                    <span style={{ fontWeight: 600 }}>{path.length}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Model Stats */}
            {stats && (
              <div style={{
                padding: '20px',
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '12px'
              }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#f1f5f9', fontSize: '18px', fontWeight: 600 }}>
                  Model Statistics
                </h3>
                <div style={{ 
                  fontSize: '14px',
                  color: '#cbd5e1',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px'
                }}>
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

            {/* Grid Info */}
            {gridState && (
              <div style={{
                padding: '20px',
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '12px'
              }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#f1f5f9', fontSize: '18px', fontWeight: 600 }}>
                  Current State
                </h3>
                <div style={{ 
                  fontSize: '14px',
                  color: '#cbd5e1',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Agent Position:</span>
                    <span style={{ fontWeight: 600 }}>({gridState.agent_pos[0]}, {gridState.agent_pos[1]})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Goal Position:</span>
                    <span style={{ fontWeight: 600 }}>({gridState.goal_pos[0]}, {gridState.goal_pos[1]})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Path Length:</span>
                    <span style={{ fontWeight: 600 }}>{path.length}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Manhattan Distance:</span>
                    <span style={{ fontWeight: 600 }}>{
                      Math.abs(gridState.agent_pos[0] - gridState.goal_pos[0]) +
                      Math.abs(gridState.agent_pos[1] - gridState.goal_pos[1])
                    }</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer style={{
          marginTop: '48px',
          padding: '24px',
          backgroundColor: '#1e293b',
          borderRadius: '12px',
          textAlign: 'center',
          border: '1px solid #334155'
        }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 600, color: '#f1f5f9', fontSize: '15px' }}>
            Smart City RL Navigation System
          </p>
          <p style={{ margin: 0, fontSize: '14px', color: '#94a3b8' }}>
            Deep Q-Network for Autonomous Pathfinding • Contributing to SDG 11: Sustainable Cities and Communities
          </p>
        </footer>
      </div>
    </div>
  );
}
