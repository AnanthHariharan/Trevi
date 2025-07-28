declare module 'react-plotly.js' {
  import { Component } from 'react';
  
  interface PlotProps {
    data: any[];
    layout?: any;
    config?: any;
    style?: React.CSSProperties;
    [key: string]: any;
  }
  
  export default class Plot extends Component<PlotProps> {}
}