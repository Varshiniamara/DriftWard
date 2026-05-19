const path = require('path');
const puppeteer = require('puppeteer');

// NOTE: must match this repo location
const dir = '/Users/varshiniamara/Desktop/Projects/ABB/DriftWarden_Proposal';

async function generatePdf() {
  console.log('Launching Puppeteer for unified index.html...');

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    const indexUrl = 'file://' + path.join(dir, 'index.html');

    console.log(`Loading URL: ${indexUrl}`);
    await page.goto(indexUrl, { waitUntil: 'networkidle0' });

    console.log('Generating PDF...');
    await page.pdf({
      path: path.join(dir, 'DriftWarden_Proposal.pdf'),
      format: 'A4',
      landscape: true,
      printBackground: true,
      margin: { top: '0mm', right: '0mm', bottom: '0mm', left: '0mm' }
    });

    console.log('PDF generated successfully at ' + path.join(dir, 'DriftWarden_Proposal.pdf'));
  } finally {
    await browser.close();
  }
}

generatePdf().catch(err => {
  console.error('Error compiling PDF:', err);
  process.exit(1);
});

