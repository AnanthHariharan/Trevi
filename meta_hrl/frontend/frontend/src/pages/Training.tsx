import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { trainingAPI } from '../utils/api';
import { useWebSocket } from '../utils/websocket';

interface TrainingMetrics {
  meta_loss?: number[];
  adaptation_loss?: number[];
  success_rate?: number[];
  diversity_score?: number[];
  learning_rate?: number[];
  iteration?: number[];
}

const Training: React.FC = () => {
  const [metrics, setMetrics] = useState<TrainingMetrics>({});
  const [isTraining, setIsTraining] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const { lastMessage, isConnected } = useWebSocket();

  useEffect(() => {
    fetchMetrics();
  }, []);

  useEffect(() => {
    if (lastMessage) {
      const message = JSON.parse(lastMessage);
      if (message.type === 'metrics_updated') {
        const newMetrics = message.data.metrics;
        setMetrics(prevMetrics => {
          const updated: TrainingMetrics = { ...prevMetrics };
          
          // Update each metric array
          Object.keys(newMetrics).forEach(key => {
            const typedKey = key as keyof TrainingMetrics;
            if (!updated[typedKey]) {
              (updated as any)[typedKey] = [];
            }
            (updated as any)[typedKey].push(newMetrics[key]);
            
            // Keep only last 1000 points
            if ((updated as any)[typedKey].length > 1000) {
              (updated as any)[typedKey] = (updated as any)[typedKey].slice(-1000);
            }
          });
          
          return updated;
        });
        
        setLastUpdate(new Date());
        setIsTraining(true);
      }
    }
  }, [lastMessage]);

  const fetchMetrics = async () => {
    try {
      const response = await trainingAPI.getMetrics();
      setMetrics(response.data.metrics);
      setLastUpdate(new Date(response.data.timestamp));
    } catch (error) {
      console.error('Error fetching training metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = () => {
    // In a real implementation, this would start a training job
    setIsTraining(true);
  };

  const handleStopTraining = () => {
    // In a real implementation, this would stop the training job
    setIsTraining(false);
  };

  const renderMetricCard = (title: string, value: string | number, trend?: string, color: string = 'primary') => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="text.secondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ color: `${color}.main` }}>
              {value}
            </Typography>
            {trend && (
              <Typography variant="body2" color="text.secondary">
                {trend}
              </Typography>
            )}
          </Box>
          <TrendingUpIcon sx={{ color: `${color}.main`, opacity: 0.7 }} />
        </Box>
      </CardContent>
    </Card>
  );

  const renderLossPlot = () => {
    if (!metrics.meta_loss || metrics.meta_loss.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No loss data available</Typography>
        </Box>
      );
    }

    const traces = [];
    
    if (metrics.meta_loss) {
      traces.push({
        x: metrics.meta_loss.map((_, index) => index),
        y: metrics.meta_loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Meta Loss',
        line: { color: '#1976d2', width: 2 },
      });
    }
    
    if (metrics.adaptation_loss) {
      traces.push({
        x: metrics.adaptation_loss.map((_, index) => index),
        y: metrics.adaptation_loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Adaptation Loss',
        line: { color: '#dc004e', width: 2 },
        yaxis: 'y2',
      });
    }

    const layout = {
      title: 'Training Loss Curves',
      xaxis: { 
        title: 'Iteration',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'Meta Loss',
        color: '#ffffff',
        gridcolor: '#444',
        type: 'log' as const,
      },
      yaxis2: {
        title: 'Adaptation Loss',
        overlaying: 'y',
        side: 'right' as const,
        color: '#ffffff',
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 60, b: 50, l: 60 },
      legend: { x: 0, y: 1 },
    };

    return (
      <Plot
        data={traces}
        layout={layout}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  const renderSuccessRatePlot = () => {
    if (!metrics.success_rate || metrics.success_rate.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No success rate data available</Typography>
        </Box>
      );
    }

    const trace = {
      x: metrics.success_rate.map((_, index) => index),
      y: metrics.success_rate.map(rate => rate * 100),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Success Rate',
      line: { color: '#2e7d32', width: 3 },
      marker: { size: 4 },
      fill: 'tonexty',
      fillcolor: 'rgba(46, 125, 50, 0.1)',
    };

    const layout = {
      title: 'Success Rate Evolution',
      xaxis: { 
        title: 'Iteration',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'Success Rate (%)',
        color: '#ffffff',
        gridcolor: '#444',
        range: [0, 100],
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
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

  const renderDiversityPlot = () => {
    if (!metrics.diversity_score || metrics.diversity_score.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No diversity data available</Typography>
        </Box>
      );
    }

    const trace = {
      x: metrics.diversity_score.map((_, index) => index),
      y: metrics.diversity_score,
      type: 'scatter',
      mode: 'lines',
      name: 'Skill Diversity',
      line: { color: '#9c27b0', width: 2 },
    };

    const layout = {
      title: 'Skill Diversity Over Time',
      xaxis: { 
        title: 'Iteration',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'Diversity Score',
        color: '#ffffff',
        gridcolor: '#444',
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
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

  const renderTrainingTable = () => {
    const recentMetrics = [];
    const length = Math.min(10, metrics.meta_loss?.length || 0);
    
    for (let i = length - 1; i >= 0; i--) {
      recentMetrics.push({
        iteration: (metrics.iteration?.[metrics.iteration.length - 1 - i] || (length - 1 - i)),
        metaLoss: metrics.meta_loss?.[metrics.meta_loss.length - 1 - i]?.toFixed(4) || 'N/A',
        adaptationLoss: metrics.adaptation_loss?.[metrics.adaptation_loss.length - 1 - i]?.toFixed(4) || 'N/A',
        successRate: metrics.success_rate?.[metrics.success_rate.length - 1 - i] ? 
          `${(metrics.success_rate[metrics.success_rate.length - 1 - i] * 100).toFixed(1)}%` : 'N/A',
        diversityScore: metrics.diversity_score?.[metrics.diversity_score.length - 1 - i]?.toFixed(3) || 'N/A',
      });
    }

    return (
      <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>Iteration</TableCell>
              <TableCell align="right">Meta Loss</TableCell>
              <TableCell align="right">Adaptation Loss</TableCell>
              <TableCell align="right">Success Rate</TableCell>
              <TableCell align="right">Diversity</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {recentMetrics.map((row, index) => (
              <TableRow key={index} hover>
                <TableCell component="th" scope="row">
                  {row.iteration}
                </TableCell>
                <TableCell align="right">{row.metaLoss}</TableCell>
                <TableCell align="right">{row.adaptationLoss}</TableCell>
                <TableCell align="right">{row.successRate}</TableCell>
                <TableCell align="right">{row.diversityScore}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const getCurrentMetrics = () => {
    const current = {
      metaLoss: metrics.meta_loss?.[metrics.meta_loss.length - 1] || 0,
      adaptationLoss: metrics.adaptation_loss?.[metrics.adaptation_loss.length - 1] || 0,
      successRate: (metrics.success_rate?.[metrics.success_rate.length - 1] || 0) * 100,
      diversityScore: metrics.diversity_score?.[metrics.diversity_score.length - 1] || 0,
      totalIterations: metrics.meta_loss?.length || 0,
    };
    return current;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <Typography>Loading training metrics...</Typography>
      </Box>
    );
  }

  const currentMetrics = getCurrentMetrics();

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Training Dashboard
        </Typography>
        
        <Box display="flex" gap={2} alignItems="center">
          <Chip
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
            size="small"
          />
          
          <Button
            variant="contained"
            startIcon={isTraining ? <StopIcon /> : <PlayIcon />}
            onClick={isTraining ? handleStopTraining : handleStartTraining}
            color={isTraining ? 'error' : 'primary'}
          >
            {isTraining ? 'Stop Training' : 'Start Training'}
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchMetrics}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Status and Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Meta Loss',
            currentMetrics.metaLoss.toFixed(4),
            undefined,
            'primary'
          )}
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Success Rate',
            `${currentMetrics.successRate.toFixed(1)}%`,
            undefined,
            'success'
          )}
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Diversity Score',
            currentMetrics.diversityScore.toFixed(3),
            undefined,
            'secondary'
          )}
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard(
            'Iterations',
            currentMetrics.totalIterations,
            undefined,
            'info'
          )}
        </Grid>
      </Grid>

      {/* Training Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              <Typography variant="h6" gutterBottom>
                Training Status
              </Typography>
              <Box display="flex" alignItems="center" gap={2}>
                <Chip
                  label={isTraining ? 'Training Active' : 'Training Idle'}
                  color={isTraining ? 'success' : 'default'}
                  icon={isTraining ? <PlayIcon /> : <StopIcon />}
                />
                {lastUpdate && (
                  <Typography variant="body2" color="text.secondary">
                    Last update: {lastUpdate.toLocaleTimeString()}
                  </Typography>
                )}
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Charts */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              {renderLossPlot()}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              {renderSuccessRatePlot()}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              {renderDiversityPlot()}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Training History
              </Typography>
              {renderTrainingTable()}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Training;