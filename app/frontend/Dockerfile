FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/nodejs-18:1-62

WORKDIR /app

USER 0
COPY package*.json ./
RUN npm install && chown -R 1001:0 /app /opt/app-root/src/.npm
USER 1001

COPY . .

ENV DISABLE_ESLINT_PLUGIN=true
ENV NPM_CONFIG_CACHE=/tmp/.npm
RUN npm run build

USER 0
RUN npm install -g serve
USER 1001

EXPOSE 3000

CMD ["serve", "-s", "build", "-l", "tcp://0.0.0.0:3000"]
