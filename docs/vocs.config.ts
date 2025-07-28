import { defineConfig } from 'vocs'

export default defineConfig({
  title: 'HSPN',
  base: '/hspn_surrogate_models/',
  sidebar: [
    {
      text: 'Getting Started',
      link: '/getting-started',
    },
    {
      text: 'Scaling',
      link: '/scaling'
    },
    {
      text: 'Troubleshooting',
      link: '/troubleshooting'
    },
    {
        text: 'API Reference',
        link: '/api/hspn'
    },
  ]
})
