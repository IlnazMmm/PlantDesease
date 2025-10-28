import axios from "axios";

const defaultBaseUrl =
  process.env.NODE_ENV === "production" ? "" : "http://localhost:8000";

export const API_BASE_URL = process.env.REACT_APP_API_URL ?? defaultBaseUrl;

const api = axios.create({
  baseURL: API_BASE_URL,
});

export default api;
