
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

// Predict race results
document.getElementById("predict-race").addEventListener("click", () => {
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
  
  // In a real implementation, we would send data to the backend
  // For now, we'll simulate the API call with a timeout
  setTimeout(() => {
    displayResults(circuit, raceDate);
  }, 2000);
});

// Display race results
function displayResults(circuit, raceDate) {
  // Hide loader
  document.getElementById("loader").style.display = "none";
  
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
  
  // Sort by predicted finish (using a simplified algorithm for demo)
  // In a real implementation, this would be the result from the ML model
  const results = drivers.map(driver => {
    // This is a simplified prediction logic for demonstration purposes
    // In reality, this would come from the ML model
    const randomFactor = Math.random() * 0.3;
    const teamStrength = {
      "Red Bull": 0.9,
      "Mercedes": 0.85,
      "Ferrari": 0.8,
      "McLaren": 0.75,
      "Aston Martin": 0.7,
      "Alpine": 0.65,
      "RB": 0.6,
      "Williams": 0.55,
      "Haas": 0.5,
      "Kick Sauber": 0.45
    };
    
    const driverStrength = {
      "Max Verstappen": 0.95,
      "Lewis Hamilton": 0.9,
      "Charles Leclerc": 0.85,
      "Lando Norris": 0.85,
      "Fernando Alonso": 0.8
    };
    
    // Calculate predicted position using start position, team strength and driver strength
    let predictedPosition = driver.startPosition * 0.6;
    predictedPosition -= (teamStrength[driver.team] || 0.5) * 10 * randomFactor;
    predictedPosition -= (driverStrength[driver.name] || 0.7) * 5 * randomFactor;
    
    // Ensure position is within bounds
    predictedPosition = Math.max(1, Math.min(20, predictedPosition));
    
    return {
      ...driver,
      predictedPosition
    };
  });
  
  // Sort by predicted position
  results.sort((a, b) => a.predictedPosition - b.predictedPosition);
  
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
  
  results.forEach((result, index) => {
    const position = index + 1;
    const posChange = result.startPosition - position;
    let posChangeClass = "";
    let posChangePrefix = "";
    
    if (posChange > 0) {
      posChangeClass = "position-gained";
      posChangePrefix = "+";
    } else if (posChange < 0) {
      posChangeClass = "position-lost";
    }
    
    // Function to convert number to ordinal
    const ordinal = (n) => {
      const s = ["th", "st", "nd", "rd"];
      const v = n % 100;
      return n + (s[(v - 20) % 10] || s[v] || s[0]);
    };
    
    tableHTML += `
      <tr>
        <td>${ordinal(position)}</td>
        <td>${result.name}</td>
        <td>
          <div class="constructor-cell">
            <div class="team-color" style="background-color: ${teams[result.team].color}"></div>
            ${result.team}
          </div>
        </td>
        <td>${ordinal(result.startPosition)}</td>
        <td class="${posChangeClass}">${posChange !== 0 ? posChangePrefix + posChange : "0"}</td>
      </tr>
    `;
  });
  
  tableHTML += `
      </tbody>
    </table>
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
