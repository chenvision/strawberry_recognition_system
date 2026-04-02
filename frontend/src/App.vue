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
body {
  margin: 0;
  padding: 0;
  background-color: #0a0a0a; /* 匹配 Dashboard 背景 */
  overflow: hidden; /* 防止滚动条 */
}

#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 100%;
  height: 100vh;
}
</style>
