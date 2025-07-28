import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';

import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import SkillLibrary from './pages/SkillLibrary';
import HierarchicalPolicy from './pages/HierarchicalPolicy';
import Training from './pages/Training';
import Visualization from './pages/Visualization';
import { WebSocketProvider } from './utils/websocket';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <WebSocketProvider url="ws://localhost:8000/ws">
        <Router>
          <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <Sidebar 
              open={sidebarOpen} 
              onToggle={handleSidebarToggle}
            />
            <Box
              component="main"
              sx={{
                flexGrow: 1,
                p: 3,
                width: { sm: `calc(100% - ${sidebarOpen ? 280 : 60}px)` },
                ml: { sm: `${sidebarOpen ? 280 : 60}px` },
                transition: 'margin 0.3s ease-in-out',
                backgroundColor: 'background.default',
              }}
            >
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/skills" element={<SkillLibrary />} />
                <Route path="/hierarchy" element={<HierarchicalPolicy />} />
                <Route path="/training" element={<Training />} />
                <Route path="/visualization" element={<Visualization />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </WebSocketProvider>
    </ThemeProvider>
  );
};

export default App;