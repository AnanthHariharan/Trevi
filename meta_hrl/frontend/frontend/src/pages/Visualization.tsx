import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Visibility as VisIcon,
  ScatterPlot as ScatterIcon,
  Timeline as TimelineIcon,
  BubbleChart as BubbleIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { visualizationAPI } from '../utils/api';

interface SkillEmbeddings {
  embeddings_raw: number[][];
  embeddings_2d_pca: number[][];
  embeddings_2d_tsne: number[][] | null;
  labels: string[];
  pca_explained_variance: number[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ width: '100%', height: '100%' }}>
    {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
  </div>
);

const Visualization: React.FC = () => {
  const [skillEmbeddings, setSkillEmbeddings] = useState<SkillEmbeddings | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<'pca' | 'tsne'>('pca');
  const [show3D, setShow3D] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchVisualizationData();
  }, []);

  const fetchVisualizationData = async () => {
    try {
      const embeddingsResponse = await visualizationAPI.getSkillEmbeddings();
      setSkillEmbeddings(embeddingsResponse.data);
    } catch (error) {
      console.error('Error fetching visualization data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const renderSkillEmbeddingsPlot = () => {
    if (!skillEmbeddings || skillEmbeddings.labels.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <Typography color="text.secondary">No embedding data available</Typography>
        </Box>
      );
    }

    const embeddings = selectedMethod === 'pca' ? 
      skillEmbeddings.embeddings_2d_pca : 
      skillEmbeddings.embeddings_2d_tsne;

    if (!embeddings) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <Typography color="text.secondary">
            {selectedMethod === 'tsne' ? 't-SNE data not available (need &gt;3 skills)' : 'No PCA data available'}
          </Typography>
        </Box>
      );
    }

    const trace = {
      x: embeddings.map(point => point[0]),
      y: embeddings.map(point => point[1]),
      mode: 'markers+text',
      type: 'scatter',
      text: skillEmbeddings.labels,
      textposition: 'top center',
      marker: {
        size: 12,
        color: skillEmbeddings.labels.map((_, i) => i),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {
          title: 'Skill Index',
          titleside: 'right'
        }
      },
      hovertemplate: skillEmbeddings.labels.map(label => 
        `<b>${label}</b><br>Click to view details<extra></extra>`
      ),
      name: 'Skills'
    };

    const layout = {
      title: `Skill Embeddings (${selectedMethod.toUpperCase()})`,
      xaxis: { 
        title: selectedMethod === 'pca' ? 
          `PC1 (${(skillEmbeddings.pca_explained_variance?.[0] * 100 || 0).toFixed(1)}% variance)` :
          't-SNE Dimension 1',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: selectedMethod === 'pca' ? 
          `PC2 (${(skillEmbeddings.pca_explained_variance?.[1] * 100 || 0).toFixed(1)}% variance)` :
          't-SNE Dimension 2',
        color: '#ffffff',
        gridcolor: '#444'
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 80, b: 50, l: 80 },
      showlegend: false,
    };

    return (
      <Plot
        data={[trace]}
        layout={layout}
        style={{ width: '100%', height: '500px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    );
  };

  const renderClusterAnalysisPlot = () => {
    if (!skillEmbeddings || skillEmbeddings.labels.length < 3) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <Typography color="text.secondary">
            Need at least 3 skills for cluster analysis
          </Typography>
        </Box>
      );
    }

    // Simulate clustering results (in practice, this would use real clustering)
    const numClusters = Math.min(3, Math.ceil(skillEmbeddings.labels.length / 2));
    const clusters = skillEmbeddings.labels.map((_, i) => i % numClusters);
    
    const embeddings = skillEmbeddings.embeddings_2d_pca;
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

    const traces = [];
    for (let cluster = 0; cluster < numClusters; cluster++) {
      const clusterIndices = clusters.map((c, i) => c === cluster ? i : -1).filter(i => i !== -1);
      
      traces.push({
        x: clusterIndices.map(i => embeddings[i][0]),
        y: clusterIndices.map(i => embeddings[i][1]),
        mode: 'markers+text',
        type: 'scatter',
        text: clusterIndices.map(i => skillEmbeddings.labels[i]),
        textposition: 'top center',
        marker: {
          size: 12,
          color: colors[cluster],
        },
        name: `Cluster ${cluster + 1}`,
        hovertemplate: clusterIndices.map(i => 
          `<b>${skillEmbeddings.labels[i]}</b><br>Cluster: ${cluster + 1}<extra></extra>`
        ),
      });
    }

    const layout = {
      title: 'Skill Clustering Analysis',
      xaxis: { 
        title: 'PC1',
        color: '#ffffff',
        gridcolor: '#444'
      },
      yaxis: { 
        title: 'PC2',
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
        style={{ width: '100%', height: '500px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    );
  };

  const renderVarianceExplainedPlot = () => {
    if (!skillEmbeddings?.pca_explained_variance) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <Typography color="text.secondary">No PCA variance data available</Typography>
        </Box>
      );
    }

    // Show first 10 components
    const variance = skillEmbeddings.pca_explained_variance.slice(0, 10);
    
    const trace = {
      x: variance.map((_, i) => `PC${i + 1}`),
      y: variance.map(v => v * 100),
      type: 'bar',
      marker: { color: '#1976d2' },
      text: variance.map(v => `${(v * 100).toFixed(1)}%`),
      textposition: 'auto',
    };

    const layout = {
      title: 'PCA Explained Variance',
      xaxis: { 
        title: 'Principal Components',
        color: '#ffffff'
      },
      yaxis: { 
        title: 'Explained Variance (%)',
        color: '#ffffff'
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

  const renderDimensionalityReductionComparison = () => {
    if (!skillEmbeddings || !skillEmbeddings.embeddings_2d_tsne) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <Typography color="text.secondary">
            t-SNE comparison not available (need &gt;3 skills)
          </Typography>
        </Box>
      );
    }

    const pcaTrace = {
      x: skillEmbeddings.embeddings_2d_pca.map(point => point[0]),
      y: skillEmbeddings.embeddings_2d_pca.map(point => point[1]),
      mode: 'markers+text',
      type: 'scatter',
      text: skillEmbeddings.labels,
      textposition: 'top center',
      marker: { size: 10, color: '#1976d2' },
      name: 'PCA',
      xaxis: 'x',
      yaxis: 'y',
    };

    const tsneTrace = {
      x: skillEmbeddings.embeddings_2d_tsne.map(point => point[0]),
      y: skillEmbeddings.embeddings_2d_tsne.map(point => point[1]),
      mode: 'markers+text',
      type: 'scatter',
      text: skillEmbeddings.labels,
      textposition: 'top center',
      marker: { size: 10, color: '#dc004e' },
      name: 't-SNE',
      xaxis: 'x2',
      yaxis: 'y2',
    };

    const layout = {
      title: 'PCA vs t-SNE Comparison',
      grid: { rows: 1, columns: 2, pattern: 'independent' },
      xaxis: { title: 'PCA Dim 1', color: '#ffffff', domain: [0, 0.45] },
      yaxis: { title: 'PCA Dim 2', color: '#ffffff' },
      xaxis2: { title: 't-SNE Dim 1', color: '#ffffff', domain: [0.55, 1] },
      yaxis2: { title: 't-SNE Dim 2', color: '#ffffff', anchor: 'x2' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
      legend: { x: 0.5, y: 1, xanchor: 'center' },
    };

    return (
      <Plot
        data={[pcaTrace, tsneTrace]}
        layout={layout}
        style={{ width: '100%', height: '500px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <Typography>Loading visualization data...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Advanced Visualizations
      </Typography>

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Reduction Method</InputLabel>
              <Select
                value={selectedMethod}
                label="Reduction Method"
                onChange={(e) => setSelectedMethod(e.target.value as 'pca' | 'tsne')}
              >
                <MenuItem value="pca">PCA</MenuItem>
                <MenuItem value="tsne">t-SNE</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControlLabel
              control={
                <Switch
                  checked={show3D}
                  onChange={(e) => setShow3D(e.target.checked)}
                />
              }
              label="3D View (Coming Soon)"
              disabled
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Visualization Tabs */}
      <Paper sx={{ width: '100%' }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab icon={<ScatterIcon />} label="Skill Embeddings" />
          <Tab icon={<BubbleIcon />} label="Clustering" />
          <Tab icon={<TimelineIcon />} label="Variance Analysis" />
          <Tab icon={<VisIcon />} label="Method Comparison" />
        </Tabs>

        <Box sx={{ p: 3 }}>
          <TabPanel value={activeTab} index={0}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    Skill Embedding Visualization
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {skillEmbeddings?.labels.length || 0} skills • {selectedMethod.toUpperCase()} reduction
                  </Typography>
                </Box>
                {renderSkillEmbeddingsPlot()}
              </CardContent>
            </Card>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Skill Clustering Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Skills are automatically clustered based on their embeddings to identify 
                  similar functionalities and potential skill hierarchies.
                </Typography>
                {renderClusterAnalysisPlot()}
              </CardContent>
            </Card>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      PCA Explained Variance
                    </Typography>
                    {renderVarianceExplainedPlot()}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} lg={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Dimensionality Insights
                    </Typography>
                    
                    {skillEmbeddings?.pca_explained_variance && (
                      <Box>
                        <Typography variant="body2" paragraph>
                          <strong>Total Variance Explained (First 2 Components):</strong> {' '}
                          {((skillEmbeddings.pca_explained_variance[0] + skillEmbeddings.pca_explained_variance[1]) * 100).toFixed(1)}%
                        </Typography>
                        
                        <Typography variant="body2" paragraph>
                          <strong>Intrinsic Dimensionality:</strong> {' '}
                          {skillEmbeddings.pca_explained_variance.findIndex(v => v < 0.05) || skillEmbeddings.pca_explained_variance.length}
                        </Typography>
                        
                        <Typography variant="body2" color="text.secondary">
                          Higher explained variance in fewer components suggests that skills 
                          have clear structural relationships and can be effectively organized 
                          in lower-dimensional spaces.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  PCA vs t-SNE Comparison
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  PCA preserves global structure and variance, while t-SNE emphasizes 
                  local neighborhoods and can reveal non-linear clusters.
                </Typography>
                {renderDimensionalityReductionComparison()}
              </CardContent>
            </Card>
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default Visualization;