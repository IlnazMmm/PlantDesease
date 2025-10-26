import React from "react";
import Home from "./pages/Home";

const App: React.FC = () => {
  return (
    <div className="app">
      <div className="app__inner">
        <h1 className="app__title">Plant Disease Detector</h1>
        <p className="app__subtitle">
          Upload a plant leaf photo to let the model predict the disease and suggested treatment.
        </p>
        <Home />
      </div>
    </div>
  );
};

export default App;
