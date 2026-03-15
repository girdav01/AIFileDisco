const fs = require('fs');
const path = require('path');

function preprocessData(inputPath, outputPath) {
  const raw = fs.readFileSync(inputPath, 'utf-8');
  const lines = raw.split('\n').filter(l => l.trim());
  const processed = lines.map(JSON.parse).map(item => ({
    text: item.text.toLowerCase().trim(),
    label: item.label
  }));
  fs.writeFileSync(outputPath, processed.map(JSON.stringify).join('\n'));
}
