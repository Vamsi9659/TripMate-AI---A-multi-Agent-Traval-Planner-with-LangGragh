(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("travel-form");
  const submitButton = $("submit-btn");
  const resultSection = $("result-section");
  const errorBox = $("error");
  let activeThreadId = null;
  let latestPlanText = "";
  const unavailable = "Not available";
  const currency = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0
  });

  function showError(message) {
    errorBox.textContent = message || "Something went wrong. Please try again.";
    errorBox.hidden = false;
  }
  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }
  function setBusy(busy, label) {
    submitButton.disabled = busy;
    submitButton.setAttribute("aria-busy", String(busy));
    submitButton.classList.toggle("is-loading", busy);
    submitButton.innerHTML = busy ? "Working with the travel agents…" : 'Create my travel plan <span>→</span>';
    if (label) $("run-state").textContent = label;
  }
  function asText(value) {
    if (value === null || value === undefined || value === "") return "No information returned.";
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  function displayValue(value) {
    return value === null || value === undefined || String(value).trim() === ""
      ? unavailable
      : String(value);
  }

  function formatCurrency(amount) {
    const value = Number(amount);
    return Number.isFinite(value) && value >= 0 ? currency.format(value) : unavailable;
  }

  function formatDate(value, options = { month: "short", day: "numeric", year: "numeric" }) {
    if (!value) return "";
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isNaN(parsed.getTime())
      ? ""
      : new Intl.DateTimeFormat("en", { ...options, timeZone: "UTC" }).format(parsed);
  }

  function formatDateRange(start, end) {
    if (!start) return unavailable;
    if (!end || start === end) return formatDate(start);
    const startDate = new Date(`${start}T00:00:00Z`);
    const endDate = new Date(`${end}T00:00:00Z`);
    if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
      return unavailable;
    }
    const sameMonth = start.slice(0, 7) === end.slice(0, 7);
    const sameYear = start.slice(0, 4) === end.slice(0, 4);
    if (sameMonth) {
      return `${formatDate(start, { month: "short", day: "numeric" })}–${endDate.getUTCDate()}, ${endDate.getUTCFullYear()}`;
    }
    const startLabel = formatDate(start, sameYear
      ? { month: "short", day: "numeric" }
      : { month: "short", day: "numeric", year: "numeric" });
    const endLabel = formatDate(end);
    return `${startLabel}–${endLabel}`;
  }

  function formatScheduleTime(value) {
    return displayValue(value).replace("T", " ").replace(/Z$/, " UTC");
  }

  function formatFlightLeg(airportName, airportCode, value) {
    const time = formatScheduleTime(value);
    const airport = [airportName, airportCode].filter(Boolean).join(" (")
      + (airportName && airportCode ? ")" : "");
    return airport ? `${airport} · ${time}` : time;
  }

  function makeCell(value, header = false) {
    const cell = document.createElement(header ? "th" : "td");
    if (header) cell.scope = "row";
    if (value instanceof Node) cell.append(value);
    else cell.textContent = displayValue(value);
    return cell;
  }

  function renderTable(tableId, rows, columnCount, emptyMessage) {
    const body = document.querySelector(`#${tableId} tbody`);
    body.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = columnCount;
      cell.className = "empty-state";
      cell.textContent = emptyMessage;
      row.append(cell);
      body.append(row);
      return;
    }
    for (const cells of rows) {
      const row = document.createElement("tr");
      for (const value of cells) row.append(makeCell(value));
      body.append(row);
    }
  }

  function renderOverview(overview = {}) {
    const destination = displayValue(overview.destination);
    const origin = displayValue(overview.origin);
    const travelers = displayValue(overview.travelers);
    renderTable("overview-table", [
      ["Destination", destination],
      ["Origin", origin],
      ["Duration", displayValue(overview.duration)],
      ["Travelers", travelers],
      ["Travel Dates", formatDateRange(overview.departure_date, overview.return_date)]
    ], 2, "Trip overview is not available.");
  }

  function renderItinerary(itineraryRows) {
    const rows = Array.isArray(itineraryRows) ? itineraryRows.map((item) => [
      `Day ${item.day}`,
      `${displayValue(item.period)} · ${displayValue(item.time)}`,
      item.place,
      item.activity,
      item.duration,
      item.estimated_cost_inr === null || item.estimated_cost_inr === undefined
        ? item.cost_status
        : `${formatCurrency(item.estimated_cost_inr)} — ${item.cost_status}`,
      item.transportation
    ]) : [];
    renderTable(
      "itinerary-table",
      rows,
      7,
      "No day-wise itinerary rows were returned."
    );
  }

  function safeExternalLink(url, label, expectedHost = "") {
    try {
      const parsed = new URL(url);
      if (!["http:", "https:"].includes(parsed.protocol)) return label;
      if (expectedHost && parsed.hostname !== expectedHost) return label;
      const link = document.createElement("a");
      link.href = parsed.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = label;
      return link;
    } catch (_) {
      return label;
    }
  }

  function renderFlights(data) {
    const flights = Array.isArray(data.flight_details) ? data.flight_details : [];
    renderTable("flight-table", flights.map((flight) => [
      flight.airline,
      flight.flight_number,
      formatFlightLeg(flight.departure_airport, flight.departure_iata, flight.departure),
      formatFlightLeg(flight.arrival_airport, flight.arrival_iata, flight.arrival),
      flight.duration,
      flight.stops,
      flight.status,
      flight.price_inr === null || flight.price_inr === undefined
        ? unavailable
        : formatCurrency(flight.price_inr)
    ]), 8, "Not available — no verified flight schedule was returned.");

    const search = $("flight-search");
    search.replaceChildren();
    const url = String(data.flight_search_url || "");
    const link = safeExternalLink(url, "Search Google Flights", "www.google.com");
    if (link instanceof Node && url.includes("/travel/flights")) {
      search.append("Fares and availability are not verified. ", link);
    } else {
      search.textContent = "Flight fares are not available from the configured schedule provider.";
    }
  }

  function renderHotels(data) {
    const hotels = Array.isArray(data.hotel_details) ? data.hotel_details : [];
    renderTable("hotel-table", hotels.map((hotel) => [
      hotel.name,
      hotel.location,
      hotel.description,
      unavailable,
      unavailable,
      safeExternalLink(hotel.url, "View source (unverified)")
    ]), 6, "No relevant named property pages were returned by hotel search.");
    const search = $("hotel-search");
    search.replaceChildren();
    const url = String(data.hotel_search_url || "");
    const link = safeExternalLink(url, "Research hotels on Google", "www.google.com");
    if (link instanceof Node) {
      search.append("Search results are not verified for rates or availability. ", link);
    } else {
      search.textContent = "No hotel inventory provider is configured; rates and availability are unverified.";
    }
  }

  function unavailableWeatherRows(overview) {
    if (!overview.departure_date) return [];
    const start = new Date(`${overview.departure_date}T00:00:00Z`);
    const end = new Date(`${(overview.return_date || overview.departure_date)}T00:00:00Z`);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end < start) return [];
    const rows = [];
    for (let day = new Date(start); day <= end; day.setUTCDate(day.getUTCDate() + 1)) {
      rows.push({
        date: day.toISOString().slice(0, 10),
        location: overview.destination,
        temperature_c: null,
        temperature_min_c: null,
        temperature_max_c: null,
        condition: "",
        rain_probability_percent: null,
        humidity: null,
        recommendation: "No reliable forecast is available for this date.",
        status: "Forecast unavailable"
      });
    }
    return rows;
  }

  function renderWeather(data, overview) {
    const forecast = Array.isArray(data.weather_forecast) && data.weather_forecast.length
      ? data.weather_forecast
      : unavailableWeatherRows(overview);
    renderTable("weather-table", forecast.map((day) => [
      formatDate(day.date),
      day.location || overview.destination,
      day.temperature_min_c === null || day.temperature_min_c === undefined
        ? unavailable
        : `${day.temperature_min_c} °C`,
      day.temperature_max_c === null || day.temperature_max_c === undefined
        ? unavailable
        : `${day.temperature_max_c} °C`,
      day.condition,
      day.rain_probability_percent === null || day.rain_probability_percent === undefined
        ? unavailable
        : `${day.rain_probability_percent}%`,
      day.humidity === null || day.humidity === undefined
        ? unavailable
        : `${day.humidity}%`,
      day.recommendation || day.status
    ]), 8, "Travel dates were not supplied; no date-specific forecast is available.");
  }

  function renderBudget(data) {
    const budget = data.budget_table || {};
    const items = Array.isArray(budget.items) ? budget.items : [];
    renderTable("budget-table", items.map((item) => [
      item.category,
      item.estimated_cost_inr === null || item.estimated_cost_inr === undefined
        ? unavailable
        : formatCurrency(item.estimated_cost_inr),
      [item.cost_status, item.price_status, item.assumption]
        .filter(Boolean)
        .join(" — ")
    ]), 3, "Budget details are not available.");
    $("budget-total").textContent = budget.estimated_total_inr === null
      || budget.estimated_total_inr === undefined
      ? unavailable
      : formatCurrency(budget.estimated_total_inr);
    const status = budget.total_status
      || "Estimated trip total unavailable until all categories have estimates.";
    const ceiling = budget.budget_ceiling_inr === null
      || budget.budget_ceiling_inr === undefined
      ? budget.budget_ceiling_status
      : `Budget ceiling ${formatCurrency(budget.budget_ceiling_inr)} — ${budget.budget_ceiling_status}`;
    $("budget-total-status").textContent = ceiling && !ceiling.startsWith("No budget ceiling")
      ? `${status} ${ceiling}`
      : status;
  }

  function renderResult(data) {
    resultSection.hidden = false;
    const itinerary = data.itinerary || data.answer || data.final_response;
    $("answer").textContent = asText(itinerary);
    $("flight-results").textContent = asText(data.flight_results);
    $("hotel-results").textContent = asText(data.hotel_results);
    $("weather-results").textContent = asText(data.weather_results);
    $("budget-results").textContent = asText(data.budget_breakdown || data.budget_results);
    const overview = data.trip_overview || {};
    renderOverview(overview);
    renderItinerary(data.itinerary_table);
    renderFlights(data);
    renderHotels(data);
    renderWeather(data, overview);
    renderBudget(data);
    const agents = Array.isArray(data.selected_agents) ? data.selected_agents : [];
    $("agent-list").replaceChildren(...agents.map((name) => {
      const chip = document.createElement("span");
      chip.className = "agent-chip";
      chip.textContent = String(name).replaceAll("_", " ");
      return chip;
    }));
    activeThreadId = data.thread_id || activeThreadId;
    latestPlanText = $("answer").textContent;
    $("approval-panel").hidden = !data.requires_approval;
    $("run-state").textContent = data.requires_approval ? "Awaiting your approval" : "Response ready";
  }
  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload)
    });
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error(`Server returned an invalid response (HTTP ${response.status}).`); }
    if (!response.ok || data.success === false) throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
    return data;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    const message = $("message").value.trim();
    if (!message) return showError("Please describe your destination and trip.");
    setBusy(true, "Agents are working");
    try {
      const payload = { message };
      if ($("departure-date").value) payload.departure_date = $("departure-date").value;
      if ($("return-date").value) payload.return_date = $("return-date").value;
      const data = await postJson("/api/travel", payload);
      renderResult(data);
      resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(false);
    }
  });

  async function submitApproval(approved) {
    clearError();
    if (!activeThreadId) return showError("No active draft is available for approval.");
    const feedback = $("feedback").value.trim();
    if (!approved && !feedback) return showError("Please enter feedback before requesting changes.");
    $("approve-btn").disabled = true;
    $("revise-btn").disabled = true;
    $("run-state").textContent = approved ? "Finalizing plan…" : "Revising draft…";
    try {
      const data = await postJson("/api/travel/approve", {
        thread_id: activeThreadId,
        approved,
        feedback
      });
      renderResult(data);
      $("feedback").value = "";
    } catch (error) {
      showError(error.message);
      $("run-state").textContent = "Action failed";
    } finally {
      $("approve-btn").disabled = false;
      $("revise-btn").disabled = false;
    }
  }
  $("approve-btn").addEventListener("click", () => submitApproval(true));
  $("revise-btn").addEventListener("click", () => submitApproval(false));
  $("copy-btn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(latestPlanText);
      $("copy-btn").textContent = "Copied ✓";
      setTimeout(() => { $("copy-btn").textContent = "Copy plan"; }, 1800);
    } catch (_) { showError("Clipboard access is unavailable in this browser."); }
  });
  $("print-btn").addEventListener("click", () => window.print());
})();
