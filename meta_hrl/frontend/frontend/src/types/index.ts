// Common types for the Meta-HRL Dashboard

export interface Skill {
  id: string;
  name: string;
  input_dim: number;
  output_dim: number;
  success_rate: number;
  is_trained: boolean;
  embedding?: number[];
  composition_chain?: string[];
  is_composite?: boolean;
}

export interface SkillLibraryStats {
  total_skills: number;
  avg_success_rate: number;
  min_success_rate: number;
  max_success_rate: number;
  trained_skills: number;
  composite_skills: number;
  skill_dimensions: {
    avg_input_dim: number;
    avg_output_dim: number;
  };
}

export interface CompositionGraph {
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
  total_nodes: number;
  total_edges: number;
}

export interface TrainingMetrics {
  meta_loss?: number[];
  adaptation_loss?: number[];
  success_rate?: number[];
  diversity_score?: number[];
  learning_rate?: number[];
  iteration?: number[];
  timestamp?: string;
}

export interface HierarchyStatus {
  status: 'active' | 'inactive' | 'not_initialized';
  current_goal?: number[] | null;
  skill_sequence?: string[];
  current_skill_index?: number;
  current_skill?: string | null;
  sequence_progress?: number;
  timestamp?: string;
}

export interface SkillEmbeddings {
  embeddings_raw: number[][];
  embeddings_2d_pca: number[][];
  embeddings_2d_tsne: number[][] | null;
  labels: string[];
  pca_explained_variance: number[];
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
}