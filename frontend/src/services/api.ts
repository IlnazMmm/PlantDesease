import axios from "axios";

export const API_BASE_URL = process.env.REACT_APP_API_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export default api;
