<template>
  <StrawberryDashboard />
</template>

<script>
import StrawberryDashboard from './components/StrawberryDashboard.vue'

export default {
  name: 'App',
  components: {
    StrawberryDashboard
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  background: #f1f5f9;
}

body {
  margin: 0;
  padding: 0;
  background-color: #f1f5f9;
  overflow: hidden;
}

#app {
  font-family: "HarmonyOS Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 100%;
  height: 100vh;
}
</style>
