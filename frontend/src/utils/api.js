import axios from 'axios';

const API_BASE = '/api';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function fetchDatasets() {
  const res = await api.get('/datasets');
  return res.data;
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

export async function deleteDataset(name) {
  const res = await api.delete(`/datasets/${name}`);
  return res.data;
}

export async function getDatasetSamples(name, n = 5) {
  const res = await api.get(`/datasets/${name}/samples`, { params: { n } });
  return res.data;
}

export async function getDatasetStats(name) {
  const res = await api.get(`/datasets/${name}/stats`);
  return res.data;
}

export async function fetchModels() {
  const res = await api.get('/models');
  return res.data;
}

export async function trainRewardModel(datasetName, params) {
  const res = await api.post(
    `/reward-model/train?dataset_name=${datasetName}`,
    params
  );
  return res.data;
}

export async function trainPolicy(params) {
  const res = await api.post('/policy/train', params);
  return res.data;
}

export async function getTrainingStatus(jobId) {
  const res = await api.get(`/training/${jobId}`);
  return res.data;
}

export async function getTrainingHistory(jobId) {
  const res = await api.get(`/training/${jobId}/history`);
  return res.data;
}

export async function getSystemInfo() {
  const res = await api.get('/info');
  return res.data;
}

export default api;

