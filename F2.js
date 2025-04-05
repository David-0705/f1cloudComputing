// Define teams and drivers
const teams = {
  "Red Bull": {
    color: "#0600EF", 
    drivers: ["Max Verstappen", "Sergio Perez"]
  },
  "Mercedes": {
    color: "#00D2BE", 
    drivers: ["Lewis Hamilton", "George Russell"]
  },
  "Ferrari": {
    color: "#DC0000", 
    drivers: ["Charles Leclerc", "Carlos Sainz"]
  },
  "McLaren": {
    color: "#FF8700", 
    drivers: ["Lando Norris", "Oscar Piastri"]
  },
  "Aston Martin": {
    color: "#006F62", 
    drivers: ["Fernando Alonso", "Lance Stroll"]
  },
  "Alpine": {
    color: "#0090FF", 
    drivers: ["Esteban Ocon", "Pierre Gasly"]
  },
  "RB": {
    color: "#1E41FF", 
    drivers: ["Yuki Tsunoda", "Daniel Ricciardo"]
  },
  "Haas": {
    color: "#FFFFFF", 
    drivers: ["Kevin Magnussen", "Nico Hülkenberg"]
  },
  "Kick Sauber": {
    color: "#900000", 
    drivers: ["Valtteri Bottas", "Zhou Guanyu"]
  },
  "Williams": {
    color: "#005AFF", 
    drivers: ["Alexander Albon", "Logan Sargeant"]
  }
};

// Tab navigation
const tabs = document.querySelectorAll(".tab");
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    const tabId = tab.getAttribute("data-tab");
    
    // Update active tab
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    
    // Show active content
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));
    document.getElementById(`${tabId}-tab`).classList.add("active");
  });
});

// Button navigation
document.getElementById("continue-to-drivers").addEventListener("click", () => {
  document.querySelectorAll(".tab")[1].click();
});

document.getElementById("back-to-setup").addEventListener("click", () => {
  document.querySelectorAll(".tab")[0].click();
});

document.getElementById("back-to-drivers").addEventListener("click", () => {
  document.querySelectorAll(".tab")[1].click();
});

document.getElementById("new-prediction").addEventListener("click", () => {
  document.querySelectorAll(".tab")[0].click();
});

// Populate driver entries
function populateDriverEntries() {
  const driverContainer = document.getElementById("driver-entries");
  driverContainer.innerHTML = "";
  
  let driverNumber = 1;
  
  for (const team in teams) {
    teams[team].drivers.forEach(driver => {
      const driverEntry = document.createElement("div");
      driverEntry.className = "driver-entry";
      driverEntry.innerHTML = `
        <div class="driver-header">
          <div>
            <div class="constructor-cell">
              <div class="team-color" style="background-color: ${teams[team].color}"></div>
              <strong>${driver}</strong>
            </div>
            <div>${team}</div>
          </div>
          <div class="driver-number">${driverNumber}</div>
        </div>
        <div class="form-group">
          <label for="position-${driverNumber}">Starting Position</label>
          <input type="number" id="position-${driverNumber}" min="1" max="20" value="${driverNumber}" 
                 data-driver="${driver}" data-team="${team}">
        </div>
      `;
      
      driverContainer.appendChild(driverEntry);
      driverNumber++;
    });
  }
}

// Load current grid button
document.getElementById("load-current-grid").addEventListener("click", () => {
  populateDriverEntries();
});

// Random grid button
document.getElementById("random-grid").addEventListener("click", () => {
  populateDriverEntries();
  
  // Create array of positions 1-20
  const positions = Array.from({length: 20}, (_, i) => i + 1);
  
  // Shuffle positions
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [positions[i], positions[j]] = [positions[j], positions[i]];
  }
  
  // Assign shuffled positions to drivers
  const inputs = document.querySelectorAll("[id^='position-']");
  inputs.forEach((input, index) => {
    input.value = positions[index];
  });
});

// API URL for predictions
const API_URL = "http://localhost:5000/api/predict";

// Predict race results
document.getElementById("predict-race").addEventListener("click", async () => {
  // Show results tab
  document.querySelectorAll(".tab")[2].click();
  
  // Show loader
  document.getElementById("loader").style.display = "block";
  document.getElementById("results-container").innerHTML = "";
  
  // Get circuit and date info
  const circuit = document.getElementById("circuit").value;
  const raceDate = document.getElementById("race-date").value;
  
  // Display circuit info
  document.getElementById("circuit-info").innerHTML = `
    <p><strong>Circuit:</strong> ${circuit}</p>
    <p><strong>Race Date:</strong> ${raceDate}</p>
  `;
  
  try {
    // Get all driver inputs
    const inputs = document.querySelectorAll("[id^='position-']");
    const drivers = [];
    
    inputs.forEach(input => {
      drivers.push({
        name: input.getAttribute("data-driver"),
        team: input.getAttribute("data-team"),
        startPosition: parseInt(input.value)
      });
    });
    
    // Call API with driver data
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        circuit: circuit,
        raceDate: raceDate,
        drivers: drivers
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API returned status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.success) {
      displayResultsFromAPI(data.results);
    } else {
      throw new Error(data.error || "Unknown error from API");
    }
  } catch (error) {
    console.error("Error predicting race results:", error);
    document.getElementById("loader").style.display = "none";
    document.getElementById("results-container").innerHTML = `
      <div class="error-message">
        Error predicting race results: ${error.message}
        <p>Make sure the Flask API is running at ${API_URL}</p>
      </div>
    `;
  }
});

// Display race results from API
function displayResultsFromAPI(results) {
  // Hide loader
  document.getElementById("loader").style.display = "none";
  
  // Create results table
  const resultsContainer = document.getElementById("results-container");
  let tableHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>Pos</th>
          <th>Driver</th>
          <th>Constructor</th>
          <th>Grid</th>
          <th>Δ Pos</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  results.forEach(result => {
    let posChangeClass = "";
    let posChangePrefix = "";
    
    if (result.positionChange > 0) {
      posChangeClass = "position-gained";
      posChangePrefix = "+";
    } else if (result.positionChange < 0) {
      posChangeClass = "position-lost";
    }
    
    tableHTML += `
      <tr>
        <td>${result.positionOrdinal}</td>
        <td>${result.driver}</td>
        <td>
          <div class="constructor-cell">
            <div class="team-color" style="background-color: ${teams[result.constructor].color}"></div>
            ${result.constructor}
          </div>
        </td>
        <td>${result.startPositionOrdinal}</td>
        <td class="${posChangeClass}">${result.positionChange !== 0 ? posChangePrefix + result.positionChange : "0"}</td>
      </tr>
    `;
  });
  
  tableHTML += `
      </tbody>
    </table>
    <div class="model-info">
      <p>Predictions made using TensorFlow neural network trained on historical F1 data</p>
    </div>
  `;
  
  resultsContainer.innerHTML = tableHTML;
}

// Initialize
window.addEventListener("DOMContentLoaded", () => {
  // Set today's date as default
  const today = new Date();
  document.getElementById("race-date").value = today.toISOString().split("T")[0];
  
  // Populate driver entries
  populateDriverEntries();
});
