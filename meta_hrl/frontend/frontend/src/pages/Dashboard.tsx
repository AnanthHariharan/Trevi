import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Psychology as SkillsIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { useWebSocket } from '../utils/websocket';
import { api } from '../utils/api';

interface DashboardStats {
  total_skills: number;
  avg_success_rate: number;
  trained_skills: number;
  composite_skills: number;
}

interface MetricCard {
  title: string;
  value: string | number;
  change?: string;
  icon: React.ReactNode;
  color: string;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [metrics, setMetrics] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const { lastMessage, isConnected } = useWebSocket();

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    if (lastMessage) {
      const message = JSON.parse(lastMessage);
      if (message.type === 'stats_update') {
        setStats(message.data);
      } else if (message.type === 'metrics_updated') {
        setMetrics(message.data.metrics);
      }
    }
  }, [lastMessage]);

  const fetchDashboardData = async () => {
    try {
      const [statsResponse, metricsResponse] = await Promise.all([
        api.get('/skill-library/stats'),
        api.get('/training/metrics'),
      ]);
      
      setStats(statsResponse.data);
      setMetrics(metricsResponse.data.metrics);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const metricCards: MetricCard[] = [
    {
      title: 'Total Skills',
      value: stats?.total_skills || 0,
      icon: <SkillsIcon />,
      color: '#1976d2',
    },
    {
      title: 'Avg Success Rate',
      value: `${((stats?.avg_success_rate || 0) * 100).toFixed(1)}%`,
      icon: <TrendingUpIcon />,
      color: '#2e7d32',
    },
    {
      title: 'Trained Skills',
      value: stats?.trained_skills || 0,
      icon: <SpeedIcon />,
      color: '#ed6c02',
    },
    {
      title: 'Composite Skills',
      value: stats?.composite_skills || 0,
      icon: <MemoryIcon />,
      color: '#9c27b0',
    },
  ];

  const renderMetricCard = (metric: MetricCard) => (
    <Grid item xs={12} sm={6} md={3} key={metric.title}>
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box>
              <Typography color="textSecondary" gutterBottom variant="overline">
                {metric.title}
              </Typography>
              <Typography variant="h4" component="div" sx={{ color: metric.color }}>
                {metric.value}
              </Typography>
              {metric.change && (
                <Typography variant="body2" color="textSecondary">
                  {metric.change}
                </Typography>
              )}
            </Box>
            <Box sx={{ color: metric.color, opacity: 0.7 }}>
              {metric.icon}
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Grid>
  );

  const renderLossPlot = () => {
    if (!metrics.meta_loss || metrics.meta_loss.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="textSecondary">No training data available</Typography>
        </Box>
      );
    }

    const data = [
      {
        x: metrics.meta_loss.map((_: any, index: number) => index),
        y: metrics.meta_loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Meta Loss',
        line: { color: '#1976d2', width: 2 },
      },
    ];

    const layout = {
      title: 'Training Loss Over Time',
      xaxis: { title: 'Iteration', color: '#ffffff' },
      yaxis: { title: 'Loss', color: '#ffffff' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
      showlegend: false,
    };

    return (
      <Plot
        data={data}
        layout={layout}
        style={{ width: '100%', height: '300px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  const renderSuccessRatePlot = () => {
    if (!stats) return null;

    const data = [
      {
        x: ['Skills'],
        y: [stats.avg_success_rate * 100],
        type: 'bar',
        marker: { color: '#2e7d32' },
      },
    ];

    const layout = {
      title: 'Average Success Rate',
      xaxis: { title: '', color: '#ffffff' },
      yaxis: { title: 'Success Rate (%)', color: '#ffffff', range: [0, 100] },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
      showlegend: false,
    };

    return (
      <Plot
        data={data}
        layout={layout}
        style={{ width: '100%', height: '300px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <LinearProgress sx={{ width: '50%' }} />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <Chip
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
            size="small"
          />
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Metric Cards */}
        {metricCards.map(renderMetricCard)}

        {/* Training Loss Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              {renderLossPlot()}
            </CardContent>
          </Card>
        </Grid>

        {/* Success Rate Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              {renderSuccessRatePlot()}
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography>WebSocket Connection</Typography>
                  <Chip
                    label={isConnected ? 'Active' : 'Inactive'}
                    color={isConnected ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Training Status</Typography>
                  <Chip
                    label={metrics.meta_loss ? 'Training' : 'Idle'}
                    color={metrics.meta_loss ? 'warning' : 'default'}
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Skill Library</Typography>
                  <Chip
                    label={`${stats?.total_skills || 0} Skills`}
                    color="info"
                    size="small"
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Typography variant="body2" color="textSecondary">
                  • Skill library updated with {stats?.total_skills || 0} skills
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  • {stats?.trained_skills || 0} skills successfully trained
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  • {stats?.composite_skills || 0} composite skills discovered
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  • Average success rate: {((stats?.avg_success_rate || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;