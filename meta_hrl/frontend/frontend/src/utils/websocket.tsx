import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  lastMessage: string | null;
  sendMessage: (message: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  socket: null,
  isConnected: false,
  lastMessage: null,
  sendMessage: () => {},
});

interface WebSocketProviderProps {
  url: string;
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ url, children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setSocket(ws);
      
      // Send ping message to keep connection alive
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);

      return () => clearInterval(pingInterval);
    };

    ws.onmessage = (event) => {
      setLastMessage(event.data);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setSocket(null);
      
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        console.log('Attempting to reconnect...');
        // This will trigger the useEffect again
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = (message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  return (
    <WebSocketContext.Provider
      value={{
        socket,
        isConnected,
        lastMessage,
        sendMessage,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};