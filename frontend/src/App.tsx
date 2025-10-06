import React from "react";
import Home from "./pages/Home";

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Plant Disease Detector</h1>
        <Home />
      </div>
    </div>
  );
};

export default App;
