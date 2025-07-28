import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API endpoints
export const skillsAPI = {
  getAll: () => api.get('/skills'),
  getById: (id: string) => api.get(`/skills/${id}`),
  create: (skillData: any) => api.post('/skills', skillData),
  getStats: () => api.get('/skill-library/stats'),
  getSimilarityMatrix: () => api.get('/skill-library/similarity-matrix'),
};

export const hierarchyAPI = {
  getStatus: () => api.get('/hierarchy/status'),
};

export const trainingAPI = {
  getMetrics: () => api.get('/training/metrics'),
  updateMetrics: (metrics: any) => api.post('/training/update-metrics', metrics),
};

export const visualizationAPI = {
  getCompositionGraph: () => api.get('/composition-graph'),
  getSkillEmbeddings: () => api.get('/visualization/skill-embeddings'),
};

export default api;