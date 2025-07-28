import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  AccountTree as HierarchyIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { hierarchyAPI } from '../utils/api';
import { useWebSocket } from '../utils/websocket';

interface HierarchyStatus {
  status: string;
  current_goal: number[] | null;
  skill_sequence: string[];
  current_skill_index: number;
  current_skill: string | null;
  sequence_progress: number;
  timestamp: string;
}

const HierarchicalPolicy: React.FC = () => {
  const [hierarchyStatus, setHierarchyStatus] = useState<HierarchyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [goalHistory, setGoalHistory] = useState<number[][]>([]);
  const [skillHistory, setSkillHistory] = useState<string[]>([]);
  const { lastMessage, isConnected } = useWebSocket();

  useEffect(() => {
    fetchHierarchyStatus();
    const interval = setInterval(fetchHierarchyStatus, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (lastMessage) {
      const message = JSON.parse(lastMessage);
      if (message.type === 'hierarchy_update') {
        setHierarchyStatus(message.data);
      }
    }
  }, [lastMessage]);

  const fetchHierarchyStatus = async () => {
    try {
      const response = await hierarchyAPI.getStatus();
      setHierarchyStatus(response.data);
      
      // Update history for visualization
      if (response.data.current_goal) {
        setGoalHistory(prev => [...prev.slice(-49), response.data.current_goal]);
      }
      if (response.data.current_skill) {
        setSkillHistory(prev => [...prev.slice(-19), response.data.current_skill]);
      }
    } catch (error) {
      console.error('Error fetching hierarchy status:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderStatusCard = () => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <HierarchyIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6">Policy Status</Typography>
        </Box>
        
        <Box display="flex" alignItems="center" mb={2}>
          <Typography variant="body2" sx={{ mr: 2 }}>
            System Status:
          </Typography>
          <Chip
            label={hierarchyStatus?.status === 'active' ? 'Active' : 'Inactive'}
            color={hierarchyStatus?.status === 'active' ? 'success' : 'default'}
            icon={hierarchyStatus?.status === 'active' ? <PlayIcon /> : <PauseIcon />}
          />
        </Box>

        <Box display="flex" alignItems="center" mb={2}>
          <Typography variant="body2" sx={{ mr: 2 }}>
            Connection:
          </Typography>
          <Chip
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            size="small"
          />
        </Box>

        {hierarchyStatus?.timestamp && (
          <Typography variant="caption" color="text.secondary">
            Last updated: {new Date(hierarchyStatus.timestamp).toLocaleTimeString()}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  const renderCurrentGoalCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Current Goal
        </Typography>
        
        {hierarchyStatus?.current_goal ? (
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Goal Vector (first 5 dimensions):
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {hierarchyStatus.current_goal.slice(0, 5).map((value, index) => (
                <Chip
                  key={index}
                  label={`${index}: ${value.toFixed(3)}`}
                  variant="outlined"
                  size="small"
                />
              ))}
              {hierarchyStatus.current_goal.length > 5 && (
                <Chip
                  label={`+${hierarchyStatus.current_goal.length - 5} more`}
                  variant="outlined"
                  size="small"
                  color="secondary"
                />
              )}
            </Box>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No active goal
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  const renderSkillSequenceCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Skill Sequence
        </Typography>
        
        {hierarchyStatus?.skill_sequence && hierarchyStatus.skill_sequence.length > 0 ? (
          <Box>
            <Box display="flex" alignItems="center" mb={2}>
              <Typography variant="body2" sx={{ mr: 2 }}>
                Progress:
              </Typography>
              <Box sx={{ width: '100%', mr: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={(hierarchyStatus.sequence_progress || 0) * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              <Typography variant="body2">
                {Math.round((hierarchyStatus.sequence_progress || 0) * 100)}%
              </Typography>
            </Box>

            <Typography variant="body2" color="text.secondary" gutterBottom>
              Sequence ({hierarchyStatus.current_skill_index + 1} of {hierarchyStatus.skill_sequence.length}):
            </Typography>
            
            <List dense>
              {hierarchyStatus.skill_sequence.map((skill, index) => (
                <ListItem
                  key={index}
                  sx={{
                    bgcolor: index === hierarchyStatus.current_skill_index ? 'action.selected' : 'transparent',
                    borderRadius: 1,
                    mb: 0.5,
                  }}
                >
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center">
                        {index < hierarchyStatus.current_skill_index ? (
                          <CheckIcon sx={{ mr: 1, color: 'success.main', fontSize: 16 }} />
                        ) : index === hierarchyStatus.current_skill_index ? (
                          <PlayIcon sx={{ mr: 1, color: 'primary.main', fontSize: 16 }} />
                        ) : (
                          <Box sx={{ width: 16, mr: 1 }} />
                        )}
                        <Typography variant="body2">
                          {skill}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No active skill sequence
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  const renderCurrentSkillCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Current Skill
        </Typography>
        
        {hierarchyStatus?.current_skill ? (
          <Box>
            <Chip
              label={hierarchyStatus.current_skill}
              color="primary"
              icon={<PlayIcon />}
              sx={{ mb: 2 }}
            />
            
            <Typography variant="body2" color="text.secondary">
              Executing skill in sequence position {hierarchyStatus.current_skill_index + 1}
            </Typography>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No skill currently executing
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  const renderGoalEvolutionPlot = () => {
    if (goalHistory.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No goal history available</Typography>
        </Box>
      );
    }

    // Plot first 3 dimensions of goal evolution
    const traces = [];
    const maxDims = Math.min(3, goalHistory[0]?.length || 0);
    
    for (let dim = 0; dim < maxDims; dim++) {
      traces.push({
        x: goalHistory.map((_, index) => index),
        y: goalHistory.map(goal => goal[dim]),
        type: 'scatter',
        mode: 'lines+markers',
        name: `Goal Dim ${dim + 1}`,
        line: { width: 2 },
      });
    }

    const layout = {
      title: 'Goal Evolution Over Time',
      xaxis: { 
        title: 'Time Steps',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'Goal Value',
        color: '#ffffff',
        gridcolor: '#444'
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
      legend: { x: 1, y: 1 },
    };

    return (
      <Plot
        data={traces}
        layout={layout}
        style={{ width: '100%', height: '300px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  const renderSkillTimelinePlot = () => {
    if (skillHistory.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No skill execution history</Typography>
        </Box>
      );
    }

    // Create skill timeline visualization
    const uniqueSkillsSet = new Set(skillHistory);
    const uniqueSkills = Array.from(uniqueSkillsSet);
    const skillToIndex: {[key: string]: number} = {};
    uniqueSkills.forEach((skill, i) => {
      skillToIndex[skill] = i;
    });

    const trace = {
      x: skillHistory.map((_, index) => index),
      y: skillHistory.map(skill => skillToIndex[skill]),
      type: 'scatter',
      mode: 'lines+markers',
      line: { shape: 'hv', width: 3 },
      marker: { size: 8 },
      hovertemplate: skillHistory.map((skill, index) => 
        `<b>Step ${index}</b><br>Skill: ${skill}<extra></extra>`
      ),
      name: 'Skill Execution',
    };

    const layout = {
      title: 'Skill Execution Timeline',
      xaxis: { 
        title: 'Time Steps',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'Skills',
        color: '#ffffff',
        gridcolor: '#444',
        tickmode: 'array',
        tickvals: uniqueSkills.map((_, i) => i),
        ticktext: uniqueSkills,
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 120 },
      showlegend: false,
    };

    return (
      <Plot
        data={[trace]}
        layout={layout}
        style={{ width: '100%', height: '300px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <Typography>Loading hierarchical policy status...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Hierarchical Policy
      </Typography>

      <Grid container spacing={3}>
        {/* Status Cards */}
        <Grid item xs={12} md={6} lg={3}>
          {renderStatusCard()}
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          {renderCurrentGoalCard()}
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          {renderCurrentSkillCard()}
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance
              </Typography>
              <Typography variant="h4" color="primary.main">
                {hierarchyStatus?.sequence_progress ? 
                  Math.round(hierarchyStatus.sequence_progress * 100) : 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Sequence completion
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Skill Sequence Details */}
        <Grid item xs={12} lg={6}>
          {renderSkillSequenceCard()}
        </Grid>

        {/* Policy Levels */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Policy Levels
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="High-Level Policy"
                    secondary="Goal setting and long-term planning"
                  />
                  <Chip 
                    label="Active" 
                    color={hierarchyStatus?.current_goal ? 'success' : 'default'} 
                    size="small" 
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Mid-Level Policy"
                    secondary="Skill selection and sequencing"
                  />
                  <Chip 
                    label="Active" 
                    color={hierarchyStatus?.skill_sequence && hierarchyStatus.skill_sequence.length > 0 ? 'success' : 'default'} 
                    size="small" 
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Low-Level Policy"
                    secondary="Primitive action execution"
                  />
                  <Chip 
                    label="Active" 
                    color={hierarchyStatus?.current_skill ? 'success' : 'default'} 
                    size="small" 
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Visualization Charts */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Goal Evolution
              </Typography>
              {renderGoalEvolutionPlot()}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Skill Execution Timeline
              </Typography>
              {renderSkillTimelinePlot()}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HierarchicalPolicy;