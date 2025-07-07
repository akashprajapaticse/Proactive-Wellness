import React, { useState, useEffect, useRef, useCallback } from 'react';

// Reusable Notification Component
const Notification = ({ id, message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(() => onClose(id), 500);
    }, 2500);
    return () => clearTimeout(timer);
  }, [id, onClose]);

  const typeClasses = {
    info: "bg-blue-600",
    warning: "bg-yellow-600",
    alert: "bg-red-600",
    success: "bg-green-600",
  };

  return (
    <div
      className={`relative p-4 rounded-lg shadow-xl text-white transition-all duration-500 ease-in-out z-50 transform
                   ${typeClasses[type] || "bg-indigo-700"}
                   ${isVisible ? "opacity-100 translate-x-0" : "opacity-0 translate-x-full pointer-events-none"}
                   mb-3 last:mb-0`}
      role="alert"
    >
      <div className="flex items-center">
        <span className="mr-3 text-lg">💡</span>
        <p className="text-base font-medium">{message}</p>
        <button
          onClick={() => setIsVisible(false)}
          className="ml-auto -mr-1 p-2 rounded-full hover:bg-white hover:bg-opacity-30 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50 transition-colors"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>
        </button>
      </div>
    </div>
  );
};

// Reusable Metric Card Component
const MetricCard = ({ icon, label, value, color }) => {
  const iconColorClass = `text-${color}-600`; // Darker icon color
  const bgColorClass = `bg-${color}-50`;
  const ringColorClass = `ring-${color}-200`;

  return (
    <div className={`flex flex-col items-center p-5 rounded-xl shadow-md ${bgColorClass} ring-1 ${ringColorClass} transform hover:scale-105 transition-all duration-200 ease-out`}>
      <MetricCard.Icon name={icon} className={`w-9 h-9 mb-3 ${iconColorClass}`} />
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <span className="text-2xl font-bold text-gray-900 mt-2">{value}</span>
    </div>
  );
};

// SVG Icon Component (utility for MetricCard and others)
MetricCard.Icon = ({ name, className }) => {
  let svgPath = '';
  let viewBox = '0 0 24 24';
  switch (name) {
    case 'heart': svgPath = "M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"; break;
    case 'mouse-pointer': svgPath = "M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"; break;
    case 'keyboard': svgPath = "M10 20H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6M10 12H7M12 12h.01M14 12h2M5 16h14M8 8h.01M12 8h.01M16 8h.01M19 8h.01"; break;
    case 'eye': svgPath = "M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"; break;
    case 'smile': svgPath = "M17 18a2 2 0 0 0-2-2H9a2 2 0 0 0-2 2M22 12A10 10 0 1 1 12 2a10 10 0 0 1 10 10zM10 10a.01.01 0 1 0 0-1 .01.01 0 0 0 0 1zM14 10a.01.01 0 1 0 0-1 .01.01 0 0 0 0 1z"; break;
    case 'user-check': svgPath = "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M15 11V5l4 6h-4M17 21h2a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2h-2v-2a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v2H9a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2"; break;
    case 'sun': svgPath = "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"; break;
    case 'monitor': svgPath = "M14 10h4a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h4M12 2h4l-2 4-2-4z"; break;
    case 'droplet': svgPath = "M12 2.69l5.66 5.66a8 8 0 1 1-11.32 0z"; break;
    case 'play': svgPath = "M5 3l14 9-14 9V3z"; break;
    case 'pause': svgPath = "M6 4h4v16H6zM14 4h4v16h-4z"; break;
    case 'square': svgPath = "M3 3h18v18H3z"; break;
    case 'walk': svgPath = "M13 14H10a4 4 0 1 0 0 8H2M16 11l-1 5 1 4M21 21l-1-4-1-4M8 8c0 1.5 0 2 1 3M12 4a3 3 0 1 1 0 6 3 3 0 0 1 0-6z"; break;
    case 'clock-alt': svgPath = "M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zM12 6v6h4"; break;
    case 'briefcase': svgPath = "M16 20V10a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM16 8V6a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2M2 20h20"; break;
    case 'zap': svgPath = "M13 2L3 14H12L11 22L21 10H12L13 2Z"; break;
    case 'moon': svgPath = "M12 3a6 6 0 0 0 9 9a9 9 0 1 1-9-9Z"; break;
    case 'battery-charging': svgPath = "M5 18H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h2M15 6h2a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2M10 12L7 7v10l3-5ZM22 10v4"; break;
    case 'cloud': svgPath = "M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"; break;
    case 'activity': svgPath = "M22 12h-4l-3 9L9 3l-3 9H2"; break;
    case 'award': svgPath = "M18 22H6a2 2 0 0 1-2-2V7l4-3 3 5 3-5 4 3v13a2 2 0 0 1-2 2zM12 9v7"; break;
    case 'coffee': svgPath = "M18 8h1a4 4 0 0 1 0 8h-1M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"; break;
    case 'book': svgPath = "M4 19.5A2.5 2.5 0 0 1 6.5 17H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15z"; break;
    case 'headphones': svgPath = "M3 14h3v2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2h3a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-3v2h3"; break;
    // New icons for specific emotions and posture
    case 'frown': svgPath = "M8 10h.01M16 10h.01M15.24 14.83a4 4 0 0 1-5.48 0M22 12A10 10 0 1 1 12 2a10 10 0 0 1 10 10z"; break;
    case 'meh': svgPath = "M9 10h.01M15 10h.01M8 14h8M22 12A10 10 0 1 1 12 2a10 10 0 0 1 10 10z"; break;
    case 'target': svgPath = "M12 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM22 12A10 10 0 1 1 12 2a10 10 0 0 1 10 10z"; break;
    case 'person-standing': svgPath = "M18 19v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2M12 4a3 3 0 1 1 0 6 3 3 0 0 1 0-6z"; break;
    case 'lamp': svgPath = "M8 2h8L12 10 8 2zM12 10v12M5 22h14"; break; // New lamp icon for lighting
    case 'ruler': svgPath = "M16 12l6 6M2 18l11-11M15 6l2-2L22 4l-3 3-5 5-2 2L6 14 2 18l4 4z"; break; // New ruler icon for distance
    default: return null;
  }
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox={viewBox}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d={svgPath} />
    </svg>
  );
};

// New Component: Time Dial Input (for HH:MM:SS with scroll/drag)
const TimeDialInput = ({ value, label, min, max, onChange, isDisabled }) => {
  const [localValue, setLocalValue] = useState(value);
  const isDragging = useRef(false);
  const startY = useRef(0);
  const startValue = useRef(0);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleValueChange = useCallback((newValue) => {
    let constrainedValue = newValue;
    if (constrainedValue < min) {
      constrainedValue = max;
    } else if (constrainedValue > max) {
      constrainedValue = min;
    }
    setLocalValue(constrainedValue);
    onChange(constrainedValue);
  }, [min, max, onChange]);

  const handleMouseMove = useCallback((e) => {
    if (isDisabled || !isDragging.current) return;
    const deltaY = e.clientY - startY.current;
    const sensitivity = 10;
    const valueChange = Math.floor(deltaY / sensitivity);

    if (valueChange !== 0) {
      const newValue = startValue.current - valueChange;
      handleValueChange(newValue);
      startY.current = e.clientY;
      startValue.current = newValue;
    }
  }, [isDisabled, handleValueChange]);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  const handleWheel = useCallback((e) => {
    if (isDisabled) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    handleValueChange(localValue + delta);
  }, [localValue, handleValueChange, isDisabled]);

  const handleMouseDown = useCallback((e) => {
    if (isDisabled) return;
    isDragging.current = true;
    startY.current = e.clientY;
    startValue.current = localValue;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [localValue, isDisabled, handleMouseMove, handleMouseUp]);

  const handleInputChange = useCallback((e) => {
    const parsedValue = parseInt(e.target.value, 10);
    if (!isNaN(parsedValue)) {
      handleValueChange(parsedValue);
    }
  }, [handleValueChange]);

  return (
    <div className="flex flex-col items-center">
      <label className="text-gray-700 text-sm mb-1">{label}</label>
      <div
        className={`w-20 h-20 bg-gray-100 rounded-lg flex items-center justify-center font-bold text-3xl cursor-pointer select-none border border-gray-300
                    ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-200 active:bg-gray-300 focus-within:ring-2 focus-within:ring-indigo-400 focus-within:border-transparent'}
                    transition-all duration-150 ease-in-out`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
      >
        {String(localValue).padStart(2, '0')}
      </div>
      <input
        type="number"
        className="sr-only"
        value={localValue}
        onChange={handleInputChange}
        min={min}
        max={max}
        disabled={isDisabled}
      />
    </div>
  );
};


// Initial state for resetting all values
const initialAppState = {
  fatigueScore: 0.0,
  heartRate: 70,
  mouseSpeed: 0.0,
  wpm: 0.0,
  gazeStability: 100.0,
  emotion: "Neutral",
  posture: "Optimal",
  lighting: "Optimal",
  distance: "Optimal",
  drinkingActionDetected: false,
  suggestion: "Welcome! Click 'Start Working Time' to begin your wellness session.",
  isAlarmActive: false,
  isLoadingMlData: false,
  notifications: [],
  waterConsumedMl: 0,
  hydrationLevel: 100,
  napTimerRunning: false,
  napTimeRemaining: 0,
  napInput: { hours: 0, minutes: 20, seconds: 0 },
  totalNapTime: 0,
  currentNapSessionDuration: 0,
  exerciseTimerRunning: false,
  exerciseTimeRemaining: 0,
  exerciseInput: { hours: 0, minutes: 5, seconds: 0 },
  totalExerciseTime: 0,
  currentExerciseSessionDuration: 0,
  workingTimeElapsed: 0,
  isWorkingTimePaused: false,
  isWorkingStarted: false,
  lastJokeIndex: -1,
  activityBackgroundColor: 'bg-gray-50',
};

// --- Constants for Recommendation Logic ---
const INACTIVITY_THRESHOLD_SECONDS = 30 * 60; // 30 minutes for prolonged inactivity alert (adjust as needed)
const SUGGESTION_SUPPRESSION_COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes for "Dismiss for a While" (adjust as needed)

// Main App Component
const App = () => {
  const [appState, setAppState] = useState(initialAppState);

  const {
    fatigueScore, heartRate, mouseSpeed, wpm, gazeStability,
    emotion, posture, lighting, distance, drinkingActionDetected,
    suggestion, isAlarmActive, isLoadingMlData, notifications,
    waterConsumedMl, hydrationLevel,
    napTimerRunning, napTimeRemaining, napInput, totalNapTime, currentNapSessionDuration,
    exerciseTimerRunning, exerciseTimeRemaining, exerciseInput, totalExerciseTime, currentExerciseSessionDuration,
    workingTimeElapsed, isWorkingTimePaused, isWorkingStarted,
    lastJokeIndex, activityBackgroundColor
  } = appState;

  // Refs for timers and notification states
  const hydrationTimerRef = useRef(null);
  const detectionTimeoutRef = useRef(null);
  const napIntervalRef = useRef(null);
  const exerciseIntervalRef = useRef(null);
  const workingIntervalRef = useRef(null);
  
  // Track last activity time for inactivity detection
  const lastSignificantActivityTime = useRef(Date.now());
  const inactivityWarningGiven = useRef(false); // Flag to ensure warning is given only once per long session

  const notifiedStates = useRef({
    heartRateHigh: false, inactivity: false, overactivity: false, gazeUnsteady: false,
    stressedFrustrated: false, boredTired: false, posturePoor: false, postureLeaning: false,
    lightingIssue: false, distanceIssue: false, hydrationLow: false, hydrationCritical: false,
    jokeShownTime: 0
  });

  const suggestionSuppressionUntil = useRef(0); // Timestamp when suggestions can resume

  const JOKE_COOLDOWN_SECONDS = 90;
  const jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Did you hear about the highly successful, yet extremely sad, biologist? He suffered from mitosis.",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "I'm reading a book about anti-gravity. It's impossible to put down!",
    "What do you call a boomerang that won’t come back? A stick!"
  ];

  const [currentTime, setCurrentTime] = useState(new Date());

  const addNotification = useCallback((message, type = "info") => {
    setAppState(prev => ({
      ...prev,
      notifications: [...prev.notifications, { id: Date.now(), message, type }]
    }));
  }, []);

  const removeNotification = useCallback((id) => {
    setAppState(prev => ({ ...prev, notifications: prev.notifications.filter(n => n.id !== id) }));
  }, []);

  const resetAllMetrics = useCallback(() => {
    clearInterval(hydrationTimerRef.current);
    clearTimeout(detectionTimeoutRef.current);
    clearInterval(napIntervalRef.current);
    clearInterval(exerciseIntervalRef.current);
    clearInterval(workingIntervalRef.current);
    hydrationTimerRef.current = null;
    detectionTimeoutRef.current = null;
    napIntervalRef.current = null;
    exerciseIntervalRef.current = null;
    workingIntervalRef.current = null;
    Object.keys(notifiedStates.current).forEach(key => { notifiedStates.current[key] = false; });
    notifiedStates.current.jokeShownTime = 0;
    lastSignificantActivityTime.current = Date.now(); // Reset activity timer
    inactivityWarningGiven.current = false; // Reset inactivity warning flag
    suggestionSuppressionUntil.current = 0; // Reset suggestion suppression
    setAppState(initialAppState);
    addNotification("All wellness metrics have been reset.", "info");
  }, [addNotification]);


  // --- ML Model Data Fetching (from Python backend via POST) ---
  const fetchMlModelOutput = useCallback(async () => {
    if (!isWorkingStarted) {
      setAppState(prev => ({ ...prev, isLoadingMlData: false }));
      return;
    }
    setAppState(prev => ({ ...prev, isLoadingMlData: true }));

    // Check for significant activity to reset inactivity timer
    // We read mouseSpeed and wpm directly from the current state (appState)
    // instead of including them in useCallback dependencies to avoid re-creating the function
    // unnecessarily and thus preventing potential infinite loops.
    // Ensure mouseSpeed and wpm are numbers before comparison
    const currentMouseSpeed = typeof appState.mouseSpeed === 'number' ? appState.mouseSpeed : 0;
    const currentWPM = typeof appState.wpm === 'number' ? appState.wpm : 0;

    if (currentMouseSpeed > 10 || currentWPM > 5) {
        lastSignificantActivityTime.current = Date.now();
        if (inactivityWarningGiven.current) { // If a warning was given, clear it as activity resumed
            inactivityWarningGiven.current = false;
            notifiedStates.current.inactivity = false; // Also clear the notification state
            addNotification("Activity resumed. Inactivity warning cleared.", "info");
        }
    }


    try {
      const response = await fetch('http://127.0.0.1:5000/predict_wellness', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hydration_level: hydrationLevel })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`HTTP error! Status: ${response.status}, Response: ${errorText}`);
        addNotification("Backend connection issue or data unavailable.", "alert");
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAppState(prev => ({
        ...prev,
        fatigueScore: typeof data.fatigue_score === 'number' ? data.fatigue_score : 0.0,
        heartRate: typeof data.heart_rate === 'number' ? data.heart_rate : 0,
        mouseSpeed: typeof data.mouse_activity === 'number' ? data.mouse_activity : 0.0,
        wpm: typeof data.keyboard_activity === 'number' ? data.keyboard_activity : 0.0,
        gazeStability: typeof data.gaze_stability === 'number' ? data.gaze_stability : 0.0,
        emotion: data.emotion || "Unknown",
        posture: data.posture || "Unknown",
        lighting: data.lighting || "Unknown",
        distance: data.distance || "Unknown",
        drinkingActionDetected: data.drinking_action_detected || false,
      }));

    } catch (error) {
      console.error("ERROR: Error fetching ML model output:", error);
    } finally {
      setAppState(prev => ({ ...prev, isLoadingMlData: false }));
    }
  }, [isWorkingStarted, hydrationLevel, addNotification, appState.mouseSpeed, appState.wpm]);


  useEffect(() => {
    let mlFetchInterval;
    if (isWorkingStarted) {
      fetchMlModelOutput();
      mlFetchInterval = setInterval(fetchMlModelOutput, 1000);
    } else {
      clearInterval(mlFetchInterval);
    }
    return () => clearInterval(mlFetchInterval);
  }, [fetchMlModelOutput, isWorkingStarted]);

  // --- Water Tracking Logic (Frontend managed) ---
  useEffect(() => {
    if (!isWorkingStarted) {
      clearInterval(hydrationTimerRef.current);
      hydrationTimerRef.current = null;
      return;
    }
    if (!hydrationTimerRef.current) {
      hydrationTimerRef.current = setInterval(() => {
        setAppState(prev => ({
          ...prev,
          hydrationLevel: Math.max(0, prev.hydrationLevel - 1)
        }));
      }, 3000); // Hydration drops every 3 seconds for demo (adjust for real use)
    }

    return () => clearInterval(hydrationTimerRef.current);
  }, [isWorkingStarted]);

  useEffect(() => {
    if (drinkingActionDetected && isWorkingStarted) {
      if (!detectionTimeoutRef.current) {
        addNotification("Drinking detected! Hydration boosted.", "success");
        setAppState(prev => ({
          ...prev,
          waterConsumedMl: prev.waterConsumedMl + 250,
          hydrationLevel: Math.min(100, prev.hydrationLevel + 20)
        }));
        detectionTimeoutRef.current = setTimeout(() => {
          detectionTimeoutRef.current = null;
        }, 3000); // Cooldown for drinking detection
      }
    }
  }, [drinkingActionDetected, isWorkingStarted, addNotification]);


  // --- Nap Countdown Timer Logic ---
  useEffect(() => {
    if (!isWorkingStarted) return;
    if (napTimerRunning && napTimeRemaining > 0) {
      napIntervalRef.current = setInterval(() => {
        setAppState(prev => ({ ...prev, napTimeRemaining: prev.napTimeRemaining - 1 }));
      }, 1000);
    } else if (napTimeRemaining === 0 && napTimerRunning) {
      clearInterval(napIntervalRef.current);
      setAppState(prev => ({
        ...prev,
        napTimerRunning: false,
        totalNapTime: prev.totalNapTime + prev.currentNapSessionDuration,
        napTimeRemaining: 0,
        currentNapSessionDuration: 0
      }));
      addNotification("Nap finished! Time to re-engage.", "success");
    }
    return () => {
        clearInterval(napIntervalRef.current);
        napIntervalRef.current = null;
    };
  }, [napTimerRunning, napTimeRemaining, napInput, isWorkingStarted, addNotification, currentNapSessionDuration]);

  const startNapTimer = () => {
    if (!isWorkingStarted) { addNotification("Please start 'Working Time' first.", "warning"); return; }
    const durationInSeconds = napInput.hours * 3600 + napInput.minutes * 60 + napInput.seconds;
    if (durationInSeconds > 0) {
        setAppState(prev => ({ ...prev, napTimeRemaining: durationInSeconds, napTimerRunning: true, currentNapSessionDuration: durationInSeconds }));
        addNotification(`Nap timer started for ${formatTime(durationInSeconds)}.`, "info");
    } else { addNotification("Please enter a valid nap duration.", "warning"); }
  };

  const stopNapTimer = () => {
    addNotification("Nap timer stopped.", "info");
    clearInterval(napIntervalRef.current);
    setAppState(prev => {
        const elapsed = prev.currentNapSessionDuration - prev.napTimeRemaining;
        return { ...prev, napTimerRunning: false, totalNapTime: prev.totalNapTime + elapsed, napTimeRemaining: 0, currentNapSessionDuration: 0 };
    });
  };

  // --- Exercise Countdown Timer Logic ---
  useEffect(() => {
    if (!isWorkingStarted) return;
    if (exerciseTimerRunning && exerciseTimeRemaining > 0) {
      exerciseIntervalRef.current = setInterval(() => {
        setAppState(prev => ({ ...prev, exerciseTimeRemaining: prev.exerciseTimeRemaining - 1 }));
      }, 1000);
    } else if (exerciseTimeRemaining === 0 && exerciseTimerRunning) {
      clearInterval(exerciseIntervalRef.current);
      setAppState(prev => ({
        ...prev,
        exerciseTimerRunning: false,
        totalExerciseTime: prev.totalExerciseTime + prev.currentExerciseSessionDuration,
        exerciseTimeRemaining: 0,
        currentExerciseSessionDuration: 0
      }));
      addNotification("Exercise finished! Good job!", "success");
    }
    return () => {
        clearInterval(exerciseIntervalRef.current);
        exerciseIntervalRef.current = null;
    };
  }, [exerciseTimerRunning, exerciseTimeRemaining, exerciseInput, isWorkingStarted, addNotification, currentExerciseSessionDuration]);

  const startExerciseTimer = () => {
    if (!isWorkingStarted) { addNotification("Please start 'Working Time' first.", "warning"); return; }
    const durationInSeconds = exerciseInput.hours * 3600 + exerciseInput.minutes * 60 + exerciseInput.seconds;
    if (durationInSeconds > 0) {
        setAppState(prev => ({ ...prev, exerciseTimeRemaining: durationInSeconds, exerciseTimerRunning: true, currentExerciseSessionDuration: durationInSeconds }));
        addNotification(`Exercise timer started for ${formatTime(durationInSeconds)}.`, "info");
    } else { addNotification("Please enter a valid exercise duration.", "warning"); }
  };

  const stopExerciseTimer = () => {
    addNotification("Exercise timer stopped.", "info");
    clearInterval(exerciseIntervalRef.current);
    setAppState(prev => {
        const elapsed = prev.currentExerciseSessionDuration - prev.exerciseTimeRemaining;
        return { ...prev, exerciseTimerRunning: false, totalExerciseTime: prev.totalExerciseTime + elapsed, exerciseTimeRemaining: 0, currentExerciseSessionDuration: 0 };
    });
  };

  // --- Working Time Pause/Resume Logic ---
  useEffect(() => {
    const shouldPauseWorkingTime = napTimerRunning || exerciseTimerRunning;
    if (!isWorkingStarted) {
      clearInterval(workingIntervalRef.current);
      workingIntervalRef.current = null;
      setAppState(prev => ({ ...prev, isWorkingTimePaused: false }));
      return;
    }
    if (shouldPauseWorkingTime && !isWorkingTimePaused) {
        setAppState(prev => ({ ...prev, isWorkingTimePaused: true }));
        if (workingIntervalRef.current) { clearInterval(workingIntervalRef.current); workingIntervalRef.current = null; }
    } else if (!shouldPauseWorkingTime && isWorkingTimePaused) {
        setAppState(prev => ({ ...prev, isWorkingTimePaused: false }));
        if (!workingIntervalRef.current) {
            workingIntervalRef.current = setInterval(() => { setAppState(prev => ({ ...prev, workingTimeElapsed: prev.workingTimeElapsed + 1 })); }, 1000);
        }
    } else if (!workingIntervalRef.current && !shouldPauseWorkingTime && !isWorkingTimePaused) {
        workingIntervalRef.current = setInterval(() => { setAppState(prev => ({ ...prev, workingTimeElapsed: prev.workingTimeElapsed + 1 })); }, 1000);
    }
    return () => { if (workingIntervalRef.current) { clearInterval(workingIntervalRef.current); workingIntervalRef.current = null; } };
  }, [napTimerRunning, exerciseTimerRunning, isWorkingTimePaused, isWorkingStarted]);

  // --- Static Clock Logic ---
  useEffect(() => {
    const clockInterval = setInterval(() => { setCurrentTime(new Date()); }, 1000);
    return () => clearInterval(clockInterval);
  }, []);

  const formatTime = (totalSeconds) => {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return [hours, minutes, seconds].map(v => v.toString().padStart(2, '0')).join(':');
  };

  // --- Dynamic Wellness Recommendation System Logic (Frontend) ---
  useEffect(() => {
    if (!isWorkingStarted) {
        setAppState(prev => ({ ...prev, suggestion: "Welcome! Click 'Start Working Time' to begin your wellness session." }));
        return;
    }

    // --- Suggestion Suppression Logic ---
    const now = Date.now();
    if (suggestionSuppressionUntil.current > now) {
        // If within suppression period, do not generate new suggestions.
        // Keep the existing suggestion text as it was when dismissed.
        return;
    }
    // If suppression period has passed, ensure the state is reset to allow new suggestions
    if (suggestionSuppressionUntil.current > 0 && suggestionSuppressionUntil.current <= now) {
        suggestionSuppressionUntil.current = 0; // Reset suppression
        setAppState(prev => ({ ...prev, suggestion: "Monitoring your wellness again. New suggestions may appear.", isAlarmActive: false }));
        // No return here, allow new suggestions to be generated in this cycle immediately after cooldown
    }


    let currentSuggestions = [];
    let alarmActive = false;

    // 1. Overall Fatigue Guidance (driven by ML-predicted fatigueScore)
    if (typeof fatigueScore === 'number' && fatigueScore >= 80) {
      currentSuggestions.push("Critical fatigue detected. IMMEDIATELY take a significant break (15-30 minutes). Consider a short nap using the timer.");
      alarmActive = true;
    } else if (typeof fatigueScore === 'number' && fatigueScore >= 60) {
      currentSuggestions.push("High fatigue detected. Take a mandatory 5-10 minute stretch or walk break. Don't push through!");
      alarmActive = true;
    } else if (typeof fatigueScore === 'number' && fatigueScore >= 40) {
      currentSuggestions.push("Moderate fatigue. A quick mental or eye break would be beneficial. Ensure proper hydration and posture.");
      alarmActive = false;
    }

    // 2. Specific Actionable Advice & Notifications (based on individual parameters)

    // Prolonged Inactivity Alert (NEW LOGIC)
    const timeSinceLastActivity = (now - lastSignificantActivityTime.current) / 1000; // in seconds
    if (timeSinceLastActivity >= INACTIVITY_THRESHOLD_SECONDS && !inactivityWarningGiven.current) {
        currentSuggestions.push(`You've been inactive for over ${INACTIVITY_THRESHOLD_SECONDS / 60} minutes. Time to stand up, stretch, or move around!`);
        addNotification(`Prolonged inactivity detected! Take a break.`, "alert");
        inactivityWarningGiven.current = true; // Set flag to prevent repeated alerts for this inactive session
        alarmActive = true;
    }


    // Heart Rate
    if (typeof heartRate === 'number' && heartRate > 95) {
        currentSuggestions.push("Your heart rate is elevated. Try deep breathing exercises, a short relaxing walk, or meditation to lower it.");
        if (!notifiedStates.current.heartRateHigh) { addNotification("Heart rate high! Focus on calming strategies.", "alert"); notifiedStates.current.heartRateHigh = true; }
    } else if (notifiedStates.current.heartRateHigh) { notifiedStates.current.heartRateHigh = false; }
    if (typeof heartRate === 'number' && heartRate < 60 && typeof fatigueScore === 'number' && fatigueScore > 30) {
        currentSuggestions.push("Heart rate is a bit low. Engage in light activity or stand up to boost circulation.");
    }

    // Activity (Mouse Speed, Typing Speed/WPM)
    // These only trigger if the long inactivity alert hasn't already covered it.
    if (typeof mouseSpeed === 'number' && mouseSpeed < 50 && typeof wpm === 'number' && wpm < 10 && timeSinceLastActivity < INACTIVITY_THRESHOLD_SECONDS) { // Check if not already in prolonged inactivity
        currentSuggestions.push("Low current activity. Ensure you're engaged. If not, consider a mental break.");
    }

    if (typeof mouseSpeed === 'number' && mouseSpeed > 400 && typeof wpm === 'number' && wpm > 80) { // High activity levels (potential for burnout)
        currentSuggestions.push("High activity levels detected. Remember to take short micro-breaks every 20-30 minutes to avoid burnout.");
        if (!notifiedStates.current.overactivity) { addNotification("Intense activity. Don't forget micro-breaks!", "info"); notifiedStates.current.overactivity = true; }
    } else if (notifiedStates.current.overactivity) { notifiedStates.current.overactivity = false; }

    // Gaze Stability (Eye Strain)
    if (typeof gazeStability === 'number' && gazeStability < 70) {
        currentSuggestions.push("Your gaze seems unsteady. Implement the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds. Blink frequently!");
        if (!notifiedStates.current.gazeUnsteady) { addNotification("Eyes tired? Try the 20-20-20 rule!", "warning"); notifiedStates.current.gazeUnsteady = true; }
        if (typeof gazeStability === 'number' && gazeStability < 50) alarmActive = true;
    } else if (notifiedStates.current.gazeUnsteady) { notifiedStates.current.gazeUnsteady = false; }

    // Emotion
    if (emotion === "Stressed" || emotion === "Frustrated") {
        currentSuggestions.push("Feeling overwhelmed? Practice mindfulness, listen to calming music, or take a short mental break.");
        if (!notifiedStates.current.stressedFrustrated) { addNotification("Stress detected! Take a moment to reset.", "alert"); notifiedStates.current.stressedFrustrated = true; }
    } else if (notifiedStates.current.stressedFrustrated) { notifiedStates.current.stressedFrustrated = false; }

    if (emotion === "Tired") {
        currentSuggestions.push("You seem tired. Consider a short nap or a caffeine-free break to refresh.");
        if ((now - notifiedStates.current.jokeShownTime) > (JOKE_COOLDOWN_SECONDS * 1000)) {
            const nextJokeIndex = (lastJokeIndex + 1) % jokes.length;
            currentSuggestions.push(`Here's a thought to perk you up: ${jokes[nextJokeIndex]}`);
            addNotification(`Here's a joke: ${jokes[nextJokeIndex]}`, "info");
            setAppState(prev => ({ ...prev, lastJokeIndex: nextJokeIndex }));
            notifiedStates.current.jokeShownTime = now;
        }
    } else if (emotion === "Bored") {
        currentSuggestions.push("Feeling bored? Try switching tasks, taking a small break, or engaging in a light activity.");
    }
    
    if (emotion === "Happy" || emotion === "Focused" || emotion === "Neutral") {
        if (typeof fatigueScore === 'number' && fatigueScore < 40 && currentSuggestions.length === 0) {
            currentSuggestions.push(`Great energy, ${emotion}! Maintain your positive work environment.`);
        }
    }

    // Posture
    if (posture === "Poor (Slumped)") {
        currentSuggestions.push("Your posture is significantly poor. Sit upright, use lumbar support, and keep feet flat on the floor.");
        if (!notifiedStates.current.posturePoor) { addNotification("Posture alert! Sit up straight.", "alert"); notifiedStates.current.posturePoor = true; }
        alarmActive = true;
    } else if (notifiedStates.current.posturePoor) { notifiedStates.current.posturePoor = false; }

    if (posture === "Leaning Forward (Strained)") {
        currentSuggestions.push("You're leaning too far forward. Adjust your monitor distance or chair position to alleviate strain.");
        if (!notifiedStates.current.postureLeaning) { addNotification("Adjust sitting position to reduce strain.", "warning"); notifiedStates.current.postureLeaning = true; }
    } else if (notifiedStates.current.postureLeaning) { notifiedStates.current.postureLeaning = false; }
    else if (posture === "Slightly Slouching") {
        currentSuggestions.push("A slight slouch is detected. Remind yourself to sit tall with shoulders relaxed and back supported.");
    }

    // Lighting
    if (lighting === "Too Dim" || lighting === "Too Bright" || lighting === "Glare") {
        currentSuggestions.push("Your lighting needs adjustment. Ensure balanced light to reduce eye strain and discomfort.");
        if (!notifiedStates.current.lightingIssue) { addNotification("Check your lighting for eye comfort.", "warning"); notifiedStates.current.lightingIssue = true; }
    } else if (notifiedStates.current.lightingIssue) { notifiedStates.current.lightingIssue = false; }

    // Distance
    if (distance === "Too Close" || distance === "Too Far") {
        currentSuggestions.push("Adjust your distance from the screen. Maintain an arm's length for optimal eye health.");
        if (!notifiedStates.current.distanceIssue) { addNotification("Adjust screen distance for optimal viewing.", "alert"); notifiedStates.current.distanceIssue = true; }
        if (distance === "Too Close") alarmActive = true;
    } else if (notifiedStates.current.distanceIssue) { notifiedStates.current.distanceIssue = false; }

    // Hydration (from frontend state)
    if (typeof hydrationLevel === 'number' && hydrationLevel < 30) {
      currentSuggestions.push("Your hydration level is very low! Please drink water immediately. Dehydration significantly impacts fatigue.");
      if (!notifiedStates.current.hydrationCritical) { addNotification("Severe dehydration detected! Drink water now.", "alert"); notifiedStates.current.hydrationCritical = true; }
      alarmActive = true;
    } else if (typeof hydrationLevel === 'number' && hydrationLevel >= 30 && notifiedStates.current.hydrationCritical) { notifiedStates.current.hydrationCritical = false; }

    if (typeof hydrationLevel === 'number' && hydrationLevel < 60 && hydrationLevel >= 30) {
      currentSuggestions.push("Your hydration level is dropping. Remember to take regular sips of water throughout your work session.");
      if (!notifiedStates.current.hydrationLow) { addNotification("Hydration dropping. Time for water!", "info"); notifiedStates.current.hydrationLow = true; }
    } else if (typeof hydrationLevel === 'number' && hydrationLevel >= 60 && notifiedStates.current.hydrationLow) { notifiedStates.current.hydrationLow = false; }

    // Final suggestion display update only if suppressed or new
    if (currentSuggestions.length === 0) {
      if (appState.suggestion !== initialAppState.suggestion || appState.isAlarmActive) { // Only clear if there was an active suggestion/alarm
          setAppState(prev => ({ ...prev, suggestion: "All systems optimal! Keep up the good work and maintain your proactive habits.", isAlarmActive: false }));
      }
    } else {
      const uniqueSuggestions = [...new Set(currentSuggestions)]; // Remove duplicates
      const newSuggestionText = uniqueSuggestions.join(" | ");
      if (newSuggestionText !== appState.suggestion) { // Only update if suggestion text has genuinely changed
        setAppState(prev => ({ ...prev, suggestion: newSuggestionText }));
      }
    }

    setAppState(prev => ({ ...prev, isAlarmActive: alarmActive }));
  }, [fatigueScore, heartRate, mouseSpeed, wpm, gazeStability, emotion, posture, lighting, distance, hydrationLevel, isWorkingStarted, addNotification, lastJokeIndex, jokes, appState.suggestion]);

  // Handle "Acknowledge & Take Action"
  const handleAcknowledgeSuggestion = useCallback(() => {
    setAppState(prev => ({
        ...prev,
        isAlarmActive: false,
        suggestion: "Suggestion acknowledged. Monitoring for new insights.",
    }));
    suggestionSuppressionUntil.current = 0; // Clear any existing suppression
    addNotification("Suggestion acknowledged. You're doing great!", "success");
    // Immediately re-evaluate suggestions to clear the acknowledged one and show new ones if applicable
    setAppState(prev => ({...prev, suggestion: " " + prev.suggestion})); // Small change to trigger useEffect
  }, [addNotification]);

  // Handle "Dismiss for a While"
  const handleDismissSuggestion = useCallback(() => {
    const suppressUntil = Date.now() + SUGGESTION_SUPPRESSION_COOLDOWN_MS;
    suggestionSuppressionUntil.current = suppressUntil;
    setAppState(prev => ({
        ...prev,
        isAlarmActive: false,
        suggestion: `Suggestions suppressed until ${new Date(suppressUntil).toLocaleTimeString()}.`,
    }));
    addNotification("Suggestions dismissed for a short period.", "info");
  }, [addNotification]);


  useEffect(() => {
    // Determine background based on highest priority (napping/exercising > alarm > fatigue)
    if (napTimerRunning) { setAppState(prev => ({ ...prev, activityBackgroundColor: 'bg-indigo-100' })); }
    else if (exerciseTimerRunning) { setAppState(prev => ({ ...prev, activityBackgroundColor: 'bg-teal-100' })); }
    else if (isAlarmActive) { setAppState(prev => ({ ...prev, activityBackgroundColor: 'bg-red-100' })); }
    else if (typeof fatigueScore === 'number' && fatigueScore >= 60) { setAppState(prev => ({ ...prev, activityBackgroundColor: 'bg-yellow-100' })); }
    else { setAppState(prev => ({ ...prev, activityBackgroundColor: 'bg-gray-50' })); }
  }, [isAlarmActive, napTimerRunning, exerciseTimerRunning, fatigueScore]); // Removed appState.isAlarmActive as it's already destructured

  const dynamicBgClasses = "bg-red-100 bg-indigo-100 bg-teal-100 bg-yellow-100 bg-gray-50 transition-colors duration-500 ease-in-out";

  return (
    <div className={`flex flex-col min-h-screen ${activityBackgroundColor} ${dynamicBgClasses} p-6 font-sans antialiased relative`}>
      {/* Hidden div to ensure Tailwind generates all dynamic color classes for metric cards */}
      <div className="hidden">
        <div className="bg-red-50 ring-red-200"></div>
        <div className="bg-blue-50 ring-blue-200"></div>
        <div className="bg-purple-50 ring-purple-200"></div>
        <div className="bg-green-50 ring-green-200"></div>
        <div className="bg-orange-50 ring-orange-200"></div>
        <div className="bg-indigo-50 ring-indigo-200"></div>
        <div className="bg-yellow-50 ring-yellow-200"></div>
        <div className="bg-teal-50 ring-teal-200"></div>
        <div className="bg-pink-50 ring-pink-200"></div>
        {/* Explicitly add text color classes for dynamic generation */}
        <div className="text-red-600 text-blue-600 text-purple-600 text-green-600 text-orange-600 text-indigo-600 text-yellow-600 text-teal-600 text-pink-600"></div>
      </div>

      {/* Notifications container */}
      <div className="fixed top-4 right-4 z-[100] p-4 flex flex-col items-end pointer-events-none">
        {notifications.map(n => (
          <Notification key={n.id} id={n.id} message={n.message} type={n.type} onClose={removeNotification} />
        ))}
      </div>

      {/* Fixed Top Section: Professional Clock and Key Timers/Hydration */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-white shadow-xl py-6 px-6 flex flex-col items-center border-b border-gray-200 rounded-b-xl">
        <div className="flex items-center justify-center space-x-4 w-full max-w-lg mb-4">
            <MetricCard.Icon name="clock-alt" className="w-12 h-12 text-gray-800" />
            <div className="text-6xl font-extrabold text-gray-900 tracking-tighter">
                {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}
            </div>
        </div>
        <div className="flex justify-center mb-6 space-x-6">
            {!isWorkingStarted ? (
                <button
                    onClick={() => setAppState(prev => ({ ...prev, isWorkingStarted: true, suggestion: "Working Time started. Monitoring your wellness." }))}
                    className="px-8 py-3 bg-indigo-700 text-white font-bold rounded-lg shadow-lg hover:bg-indigo-800 focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105"
                >
                    Start Working Time
                </button>
            ) : (
                <button
                    onClick={resetAllMetrics}
                    className="px-8 py-3 bg-red-700 text-white font-bold rounded-lg shadow-lg hover:bg-red-800 focus:outline-none focus:ring-2 focus:ring-red-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105"
                >
                    Stop Working Time (Reset All)
                </button>
            )}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 w-full max-w-6xl">
            <div className="flex flex-col items-center justify-center p-4 rounded-xl shadow-md bg-purple-50 ring-1 ring-purple-200">
                <MetricCard.Icon name="briefcase" className="w-7 h-7 mb-2 text-purple-700" />
                <span className="text-sm font-semibold text-gray-700">Working Time</span>
                <span className="text-xl font-bold text-gray-900 mt-1 font-mono">{formatTime(workingTimeElapsed)}</span>
                {isWorkingTimePaused && <span className="text-red-600 text-xs font-bold">Paused</span>}
            </div>
            <div className="flex flex-col items-center p-4 rounded-xl shadow-md bg-blue-50 ring-1 ring-blue-200 overflow-hidden relative h-[160px]">
                <MetricCard.Icon name="droplet" className="w-11 h-11 text-blue-700 z-20 mt-1" />
                <span className="text-sm font-semibold text-gray-700 z-10 mt-1">Hydration Level</span>
                <span className="text-xl font-bold text-gray-900 mt-0 font-mono z-10">{Math.round(hydrationLevel)}%</span>
                <div
                    className="absolute bottom-0 left-0 right-0 rounded-b-xl transition-all duration-500 ease-in-out z-0"
                    style={{ height: `${hydrationLevel}%`, backgroundColor: '#2563eb', opacity: 0.9 }}
                ></div>
            </div>
            <div className="flex flex-col items-center justify-center p-4 rounded-xl shadow-md bg-green-50 ring-1 ring-green-200">
                <span role="img" aria-label="nap time" className="text-3xl mb-2">😴</span>
                <span className="text-sm font-semibold text-gray-700">Nap Time</span>
                <span className="text-base font-bold text-gray-800 mt-1 font-mono">
                  {napTimerRunning ? "Current:" : "Total Napped:"}
                </span>
                <span className="text-xl font-bold text-gray-900 font-mono">
                  {napTimerRunning ? formatTime(napTimeRemaining) : formatTime(totalNapTime)}
                </span>
                {napTimerRunning && <span className="text-green-700 text-xs font-bold">Running</span>}
            </div>
            <div className="flex flex-col items-center justify-center p-4 rounded-xl shadow-md bg-orange-50 ring-1 ring-orange-200">
                <MetricCard.Icon name="walk" className="w-7 h-7 mb-2 text-orange-700" />
                <span className="text-sm font-semibold text-gray-700">Exercise Time</span>
                <span className="text-base font-bold text-gray-800 mt-1 font-mono">
                  {exerciseTimerRunning ? "Current:" : "Total Exercised:"}
                </span>
                <span className="text-xl font-bold text-gray-900 font-mono">
                  {exerciseTimerRunning ? formatTime(exerciseTimeRemaining) : formatTime(totalExerciseTime)}
                </span>
                {exerciseTimerRunning && <span className="text-orange-700 text-xs font-bold">Running</span>}
            </div>
        </div>
      </div>

      {/* Main Dashboard Area */}
      <div className="relative w-full max-w-6xl mx-auto bg-white rounded-xl shadow-2xl p-8 md:p-10 border border-gray-200 ring-1 ring-gray-100 mt-[300px]">
        <h1 className="text-4xl font-extrabold text-center text-gray-900 mb-8 tracking-tight">
          Proactive Wellness Dashboard
        </h1>

        {/* Current Metrics Section */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 mb-10">
          <MetricCard icon="heart" label="Heart Rate" value={isLoadingMlData ? "..." : `${heartRate} BPM`} color="red" />
          <MetricCard 
            icon="mouse-pointer" 
            label="Mouse Speed" 
            value={isLoadingMlData ? "..." : (typeof mouseSpeed === 'number' ? `${mouseSpeed.toFixed(2)} PPS` : 'N/A')} 
            color="blue" 
          />
          <MetricCard 
            icon="keyboard" 
            label="Typing Speed" 
            value={isLoadingMlData ? "..." : (typeof wpm === 'number' ? `${Math.round(wpm)} WPM` : 'N/A')} 
            color="purple" 
          />
          <MetricCard 
            icon="eye" 
            label="Gaze Stability" 
            value={isLoadingMlData ? "..." : (typeof gazeStability === 'number' ? `${gazeStability.toFixed(2)}/100` : 'N/A')} 
            color="green" 
          />
          {/* Dynamic Emotion Icon */}
          <MetricCard
            icon={
              emotion === "Stressed" || emotion === "Frustrated" || emotion === "Tired" ? "frown" :
              emotion === "Bored" || emotion === "Neutral" ? "meh" :
              emotion === "Focused" ? "target" :
              "smile"
            }
            label="Emotion"
            value={isLoadingMlData ? "..." : emotion}
            color="orange"
          />
          {/* Updated Posture Icon */}
          <MetricCard icon="person-standing" label="Posture" value={isLoadingMlData ? "..." : posture} color="indigo" />
          <MetricCard icon="lamp" label="Lighting" value={isLoadingMlData ? "..." : lighting} color="yellow" />
          <MetricCard icon="ruler" label="Laptop Distance" value={isLoadingMlData ? "..." : distance} color="teal" />

          {/* Fatigue Score Display */}
          <div className="flex flex-col items-center p-5 rounded-xl shadow-md bg-pink-50 ring-1 ring-pink-200 transform hover:scale-105 transition-all duration-200 ease-out">
            <MetricCard.Icon name="zap" className={`w-9 h-9 mb-3 text-pink-600`} />
            <span className="text-sm font-medium text-gray-700">Fatigue Score</span>
            <span className="text-2xl font-bold mt-1 text-gray-900">
              {isLoadingMlData ? "..." : (typeof fatigueScore === 'number' ? fatigueScore.toFixed(2) : 'N/A')}
            </span>
          </div>
        </div>

        {/* Water Management Section */}
        <div className="bg-blue-50 ring-2 ring-blue-300 rounded-xl shadow-lg p-7 mb-10 flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center mb-4 md:mb-0">
                <MetricCard.Icon name="droplet" className={`w-9 h-9 mr-3 text-blue-600`} />
                <div className="ml-4 text-gray-800">
                    <p className="text-xl font-semibold">Total Water Consumed: {waterConsumedMl} ml</p>
                    <p className="text-base">Current Hydration Level: {Math.round(hydrationLevel)}%</p>
                    <p className="text-sm text-gray-600">Stay hydrated throughout your work hours!</p>
                </div>
            </div>
            <div className="flex gap-3">
                <button
                    onClick={() => {
                        setAppState(prev => ({
                            ...prev,
                            waterConsumedMl: prev.waterConsumedMl + 250,
                            hydrationLevel: Math.min(100, prev.hydrationLevel + 20)
                        }));
                        addNotification("Manually logged 250ml water.", "info");
                    }}
                    className="px-6 py-3 bg-blue-700 text-white font-bold rounded-lg shadow-md hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105"
                >
                    Log 250ml (Manual)
                </button>
            </div>
        </div>

        {/* Nap Timer Control Section */}
        <div className="bg-green-50 ring-2 ring-green-300 rounded-xl shadow-lg p-7 mb-10">
            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
                <span className="text-green-700 mr-3 text-3xl">😴</span> Nap Timer Controls
            </h2>
            <p className="text-gray-700 leading-relaxed mb-5">
                Set a nap duration (HH:MM:SS) and start the countdown.
                Recommended duration for boosting alertness is 20-30 minutes.
            </p>
            <div className="flex flex-col md:flex-row items-center justify-center space-x-0 md:space-x-6 space-y-5 md:space-y-0 mb-4">
                <TimeDialInput label="H" value={napInput.hours} min={0} max={23} onChange={(val) => setAppState(prev => ({ ...prev, napInput: { ...prev.napInput, hours: val } }))} isDisabled={napTimerRunning || !isWorkingStarted} />
                <TimeDialInput label="M" value={napInput.minutes} min={0} max={59} onChange={(val) => setAppState(prev => ({ ...prev, napInput: { ...prev.napInput, minutes: val } }))} isDisabled={napTimerRunning || !isWorkingStarted} />
                <TimeDialInput label="S" value={napInput.seconds} min={0} max={59} onChange={(val) => setAppState(prev => ({ ...prev, napInput: { ...prev.napInput, seconds: val } }))} isDisabled={napTimerRunning || !isWorkingStarted} />
                <div className="flex space-x-3 mt-4 md:mt-0">
                    {!napTimerRunning ? (
                        <button onClick={startNapTimer} className="flex items-center px-6 py-3 bg-green-700 text-white font-bold rounded-lg shadow-md hover:bg-green-800 focus:outline-none focus:ring-2 focus:ring-green-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105" disabled={!isWorkingStarted}>
                            <MetricCard.Icon name="play" className="w-6 h-6 mr-2" /> Start Nap
                        </button>
                    ) : (
                        <button onClick={stopNapTimer} className="flex items-center px-6 py-3 bg-yellow-600 text-white font-bold rounded-lg shadow-md hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105" disabled={!isWorkingStarted}>
                            <MetricCard.Icon name="pause" className="w-6 h-6 mr-2" /> Stop Nap
                        </button>
                    )}
                </div>
            </div>
        </div>

        {/* Exercise Timer Control Section */}
        <div className="bg-orange-50 ring-2 ring-orange-300 rounded-xl shadow-lg p-7 mb-10">
            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
              <MetricCard.Icon name="walk" className="w-7 h-7 mr-3 text-orange-700" /> Exercise Timer Controls
            </h2>
            <p className="text-gray-700 leading-relaxed mb-5">
                Set an exercise duration (HH:MM:SS) and start the countdown.
                Regular short exercise breaks can significantly boost energy and focus.
            </p>
            <div className="flex flex-col md:flex-row items-center justify-center space-x-0 md:space-x-6 space-y-5 md:space-y-0 mb-4">
                <TimeDialInput label="H" value={exerciseInput.hours} min={0} max={23} onChange={(val) => setAppState(prev => ({ ...prev, exerciseInput: { ...prev.exerciseInput, hours: val } }))} isDisabled={exerciseTimerRunning || !isWorkingStarted} />
                <TimeDialInput label="M" value={exerciseInput.minutes} min={0} max={59} onChange={(val) => setAppState(prev => ({ ...prev, exerciseInput: { ...prev.exerciseInput, minutes: val } }))} isDisabled={exerciseTimerRunning || !isWorkingStarted} />
                <TimeDialInput label="S" value={exerciseInput.seconds} min={0} max={59} onChange={(val) => setAppState(prev => ({ ...prev, exerciseInput: { ...prev.exerciseInput, seconds: val } }))} isDisabled={exerciseTimerRunning || !isWorkingStarted} />
                <div className="flex space-x-3 mt-4 md:mt-0">
                    {!exerciseTimerRunning ? (
                        <button onClick={startExerciseTimer} className="flex items-center px-6 py-3 bg-orange-700 text-white font-bold rounded-lg shadow-md hover:bg-orange-800 focus:outline-none focus:ring-2 focus:ring-orange-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105" disabled={!isWorkingStarted}>
                            <MetricCard.Icon name="play" className="w-6 h-6 mr-2" /> Start Exercise
                        </button>
                    ) : (
                        <button onClick={stopExerciseTimer} className="flex items-center px-6 py-3 bg-yellow-600 text-white font-bold rounded-lg shadow-md hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105" disabled={!isWorkingStarted}>
                            <MetricCard.Icon name="pause" className="w-6 h-6 mr-2" /> Stop Exercise
                        </button>
                    )}
                </div>
            </div>
        </div>

        {/* Suggestion / Alarm Area */}
        <div className={`relative rounded-xl p-7 transition-all duration-500 ease-in-out
                           ${isAlarmActive ? 'bg-gradient-to-r from-red-100 to-orange-100 ring-4 ring-red-400 shadow-2xl' : 'bg-green-50 ring-2 ring-green-300 shadow-lg'}`}>
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
            {isAlarmActive ? (
              <span className="animate-pulse text-red-600 mr-3 text-3xl">🚨</span>
            ) : (
              <span className="text-green-700 mr-3 text-3xl">✨</span>
            )}
            Wellness Suggestion
          </h2>
          <p className="text-gray-700 leading-relaxed text-base mb-5">
            {isLoadingMlData ? "Analyzing data for new insights and recommendations..." : suggestion}
          </p>
          {(isAlarmActive || (suggestion !== initialAppState.suggestion && suggestionSuppressionUntil.current === 0)) && (
            <div className="flex flex-col md:flex-row gap-4 mt-6">
                <button
                onClick={handleAcknowledgeSuggestion}
                className="w-full md:w-auto px-7 py-3 bg-indigo-700 text-white font-bold rounded-lg shadow-md hover:bg-indigo-800 focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105"
                >
                Acknowledge & Take Action
                </button>
                <button
                onClick={handleDismissSuggestion}
                className="w-full md:w-auto px-7 py-3 bg-gray-500 text-white font-bold rounded-lg shadow-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-75 transition-all duration-200 transform hover:scale-105"
                >
                Dismiss for a While
                </button>
            </div>
          )}
          {isLoadingMlData && (
            <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-80 rounded-xl">
              <div className="flex items-center text-gray-800 font-bold text-lg">
                <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-gray-800" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing ML data...
              </div>
            </div>
          )}
        </div>

        {/* Explanation Section */}
        <div className="mt-10 text-gray-600 text-base leading-relaxed border-t border-gray-200 pt-8">
          <h3 className="text-xl font-bold text-gray-900 mb-3">How this works:</h3>
          <ul className="list-disc list-inside space-y-2">
            <li><span className="font-extrabold text-gray-800">Real-time Data Collection:</span> Your webcam (via backend Python) continuously captures eye movements, emotions, body posture, lighting, and laptop distance. Mouse and keyboard activity are also tracked via Python listeners. Heart rate is simulated.</li>
            <li><span className="font-extrabold text-gray-800">ML Fatigue Prediction:</span> The collected real-time metrics, along with your current hydration level (from this app), are sent to a Machine Learning model running in the Python backend. This model predicts your current `fatigue_score` from 0-100.</li>
            <li><span className="font-extrabold text-gray-800">Intelligent Recommendation System:</span> Based on the ML-predicted `fatigue_score` and individual metric deviations, this frontend system provides tailored, actionable advice for breaks, posture correction, eye care, stress relief, environmental adjustments, and hydration.</li>
            <li><span className="font-extrabold text-gray-800">Dynamic UI Feedback:</span> The user interface dynamically updates to reflect current metrics, provides urgent alarms for critical issues, and offers interactive tools like the water tracker and nap/exercise timers.</li>
          </ul>
          <p className="mt-5 text-sm text-gray-500">
            **Important Note:** This demonstration uses a synthetic ML model for fatigue prediction. In a real production system, this model would be trained on extensive real-world (and ethically sourced) physiological and behavioral data. Actual webcam features require user permission and significant computational resources.
          </p>
          <ul className="list-disc list-inside ml-4 mt-3 text-sm text-gray-600">
              <li>Ensure your webcam is enabled and not in use by other applications.</li>
              <li>Make sure `app.py` is running in your Python environment before starting this React app.</li>
          </ul>
          <div className="mt-5 text-center text-xs text-gray-500">
            Your current user ID (for potential data persistence with Firestore): <span className="font-mono text-gray-700">{typeof window !== 'undefined' && window.crypto.randomUUID ? window.crypto.randomUUID() : 'anonymous-user'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
