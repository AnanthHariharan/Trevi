import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemText,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  Psychology as SkillIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { skillsAPI, visualizationAPI } from '../utils/api';

interface Skill {
  id: string;
  name: string;
  input_dim: number;
  output_dim: number;
  success_rate: number;
  is_trained: boolean;
}

interface CompositionGraph {
  nodes: Array<{
    id: string;
    name: string;
    success_rate: number;
    is_composite: boolean;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relationship: string;
  }>;
}

const SkillLibrary: React.FC = () => {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [filteredSkills, setFilteredSkills] = useState<Skill[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSkill, setSelectedSkill] = useState<Skill | null>(null);
  const [compositionGraph, setCompositionGraph] = useState<CompositionGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    fetchSkills();
    fetchCompositionGraph();
  }, []);

  useEffect(() => {
    const filtered = skills.filter(skill =>
      skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      skill.id.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredSkills(filtered);
  }, [skills, searchTerm]);

  const fetchSkills = async () => {
    try {
      const response = await skillsAPI.getAll();
      setSkills(response.data.skills);
    } catch (error) {
      console.error('Error fetching skills:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchCompositionGraph = async () => {
    try {
      const response = await visualizationAPI.getCompositionGraph();
      setCompositionGraph(response.data);
    } catch (error) {
      console.error('Error fetching composition graph:', error);
    }
  };

  const handleSkillClick = (skill: Skill) => {
    setSelectedSkill(skill);
    setDialogOpen(true);
  };

  const renderSkillCard = (skill: Skill) => (
    <Grid item xs={12} sm={6} md={4} key={skill.id}>
      <Card 
        sx={{ 
          height: '100%', 
          cursor: 'pointer',
          '&:hover': { 
            transform: 'translateY(-2px)',
            boxShadow: 4,
          },
          transition: 'all 0.2s ease-in-out'
        }}
        onClick={() => handleSkillClick(skill)}
      >
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <SkillIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="div" noWrap>
              {skill.name}
            </Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            ID: {skill.id}
          </Typography>
          
          <Box display="flex" justifyContent="space-between" mb={2}>
            <Typography variant="body2">
              Input: {skill.input_dim}D
            </Typography>
            <Typography variant="body2">
              Output: {skill.output_dim}D
            </Typography>
          </Box>
          
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="body2" component="div">
              Success Rate: {(skill.success_rate * 100).toFixed(1)}%
            </Typography>
            <Chip
              label={skill.is_trained ? 'Trained' : 'Untrained'}
              color={skill.is_trained ? 'success' : 'default'}
              size="small"
            />
          </Box>
        </CardContent>
      </Card>
    </Grid>
  );

  const renderCompositionGraph = () => {
    if (!compositionGraph || compositionGraph.nodes.length === 0) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <Typography color="text.secondary">No composition data available</Typography>
        </Box>
      );
    }

    // Create network layout data for Plotly
    const nodeData = {
      x: compositionGraph.nodes.map((_, i) => Math.cos(2 * Math.PI * i / compositionGraph.nodes.length)),
      y: compositionGraph.nodes.map((_, i) => Math.sin(2 * Math.PI * i / compositionGraph.nodes.length)),
      mode: 'markers+text',
      type: 'scatter',
      text: compositionGraph.nodes.map(node => node.name),
      textposition: 'middle center',
      marker: {
        size: compositionGraph.nodes.map(node => node.is_composite ? 20 : 15),
        color: compositionGraph.nodes.map(node => node.success_rate),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {
          title: 'Success Rate',
          titleside: 'right'
        }
      },
      hovertemplate: compositionGraph.nodes.map(node => 
        `<b>${node.name}</b><br>Success Rate: ${(node.success_rate * 100).toFixed(1)}%<br>Type: ${node.is_composite ? 'Composite' : 'Primitive'}<extra></extra>`
      ),
      name: 'Skills'
    };

    // Create edge traces
    const edgeTraces = compositionGraph.edges.map(edge => {
      const sourceIdx = compositionGraph.nodes.findIndex(n => n.id === edge.source);
      const targetIdx = compositionGraph.nodes.findIndex(n => n.id === edge.target);
      
      if (sourceIdx === -1 || targetIdx === -1) return null;
      
      return {
        x: [nodeData.x[sourceIdx], nodeData.x[targetIdx], null],
        y: [nodeData.y[sourceIdx], nodeData.y[targetIdx], null],
        mode: 'lines',
        type: 'scatter',
        line: { color: '#888', width: 2 },
        hoverinfo: 'none',
        showlegend: false
      };
    }).filter(trace => trace !== null);

    const layout = {
      title: 'Skill Composition Network',
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#ffffff' },
      margin: { t: 50, r: 30, b: 30, l: 30 },
      showlegend: false,
    };

    return (
      <Plot
        data={[...edgeTraces, nodeData]}
        layout={layout}
        style={{ width: '100%', height: '400px' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  };

  const renderSkillDialog = () => (
    <Dialog 
      open={dialogOpen} 
      onClose={() => setDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box display="flex" alignItems="center">
          <SkillIcon sx={{ mr: 1 }} />
          {selectedSkill?.name}
        </Box>
      </DialogTitle>
      <DialogContent>
        {selectedSkill && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="subtitle2" gutterBottom>
                Basic Information
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Skill ID" secondary={selectedSkill.id} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Input Dimension" secondary={selectedSkill.input_dim} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Output Dimension" secondary={selectedSkill.output_dim} />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Success Rate" 
                    secondary={`${(selectedSkill.success_rate * 100).toFixed(1)}%`} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Training Status" 
                    secondary={selectedSkill.is_trained ? 'Trained' : 'Untrained'} 
                  />
                </ListItem>
              </List>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="subtitle2" gutterBottom>
                Performance Metrics
              </Typography>
              <Box sx={{ height: 200, bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Detailed performance metrics would be displayed here in a real implementation.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <Typography>Loading skills...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Skill Library
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {/* Add new skill functionality */}}
        >
          Add Skill
        </Button>
      </Box>

      {/* Search Bar */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search skills by name or ID..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
          }}
        />
      </Paper>

      {/* Stats */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Skills
              </Typography>
              <Typography variant="h4" component="div">
                {skills.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Trained Skills
              </Typography>
              <Typography variant="h4" component="div">
                {skills.filter(s => s.is_trained).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Avg Success Rate
              </Typography>
              <Typography variant="h4" component="div">
                {skills.length > 0 
                  ? (skills.reduce((acc, s) => acc + s.success_rate, 0) / skills.length * 100).toFixed(1) + '%'
                  : '0%'
                }
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Composite Skills
              </Typography>
              <Typography variant="h4" component="div">
                {compositionGraph?.edges.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Composition Graph */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Skill Composition Network
          </Typography>
          {renderCompositionGraph()}
        </CardContent>
      </Card>

      {/* Skills Grid */}
      <Typography variant="h6" gutterBottom>
        Skills ({filteredSkills.length})
      </Typography>
      <Grid container spacing={3}>
        {filteredSkills.map(renderSkillCard)}
      </Grid>

      {filteredSkills.length === 0 && searchTerm && (
        <Box display="flex" justifyContent="center" alignItems="center" height={200}>
          <Typography color="text.secondary">
            No skills found matching "{searchTerm}"
          </Typography>
        </Box>
      )}

      {renderSkillDialog()}
    </Box>
  );
};

export default SkillLibrary;