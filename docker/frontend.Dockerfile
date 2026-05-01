# medkit Vite dev server.
FROM node:22-alpine

WORKDIR /app

# Install deps first for layer caching.
COPY package.json package-lock.json /app/
RUN npm ci --no-audit --no-fund

COPY . /app

EXPOSE 5173

# Bind to all interfaces so docker port-forwarding works.
CMD ["node", "node_modules/vite/bin/vite.js", "--host", "0.0.0.0", "--port", "5173"]
