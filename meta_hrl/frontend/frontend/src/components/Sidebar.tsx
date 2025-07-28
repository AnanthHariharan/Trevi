import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Box,
  Typography,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology as SkillsIcon,
  AccountTree as HierarchyIcon,
  TrendingUp as TrainingIcon,
  Visibility as VisualizationIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  Memory as MemoryIcon,
} from '@mui/icons-material';

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
}

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Skill Library', icon: <SkillsIcon />, path: '/skills' },
  { text: 'Hierarchical Policy', icon: <HierarchyIcon />, path: '/hierarchy' },
  { text: 'Training', icon: <TrainingIcon />, path: '/training' },
  { text: 'Visualization', icon: <VisualizationIcon />, path: '/visualization' },
];

const Sidebar: React.FC<SidebarProps> = ({ open, onToggle }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const drawerWidth = open ? 280 : 60;

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#1a1a1a',
          borderRight: '1px solid #333',
          transition: 'width 0.3s ease-in-out',
        },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          padding: 2,
          minHeight: 64,
        }}
      >
        {open && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <MemoryIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
              Meta-HRL
            </Typography>
          </Box>
        )}
        <IconButton onClick={onToggle} sx={{ color: 'text.primary' }}>
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>
      
      <Divider sx={{ borderColor: '#333' }} />
      
      <List sx={{ flexGrow: 1, pt: 2 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ display: 'block', mb: 1 }}>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={location.pathname === item.path}
              sx={{
                minHeight: 48,
                justifyContent: open ? 'initial' : 'center',
                px: 2.5,
                mx: 1,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                },
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: open ? 3 : 'auto',
                  justifyContent: 'center',
                  color: location.pathname === item.path ? 'white' : 'text.primary',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                sx={{
                  opacity: open ? 1 : 0,
                  color: location.pathname === item.path ? 'white' : 'text.primary',
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      {open && (
        <>
          <Divider sx={{ borderColor: '#333' }} />
          <Box sx={{ p: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Meta-Learning for Hierarchical Skill Acquisition v1.0.0
            </Typography>
          </Box>
        </>
      )}
    </Drawer>
  );
};

export default Sidebar;