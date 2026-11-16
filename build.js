// Inline engine.js into template.html → index.html (single-file page). Usage: node build.js
const fs = require('fs'); const t = fs.readFileSync(__dirname + '/template.html', 'utf8'); const e = fs.readFileSync(__dirname + '/engine.js', 'utf8');
if (e.includes('</script')) throw new Error('engine.js must not contain </script');
fs.writeFileSync(__dirname + '/index.html', t.replace('/*__ENGINE__*/', () => e)); console.log('index.html built');
