import axios from 'axios';
import { getApiBase } from './apiBase';

export const apiClient = axios.create({
  baseURL: getApiBase(),
  timeout: 15000
});

