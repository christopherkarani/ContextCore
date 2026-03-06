/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core-concepts/four-tier-memory',
        'core-concepts/token-budgeting',
        'core-concepts/sessions',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/configuring-agent-context',
        'guides/building-context-windows',
        'guides/memory-management',
        'guides/progressive-compression',
        'guides/session-persistence',
        'guides/custom-embedding-providers',
        'guides/custom-token-counters',
      ],
    },
    {
      type: 'category',
      label: 'Metal Engine',
      items: [
        'metal-engine/gpu-scoring',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api-reference/agent-context',
        'api-reference/context-configuration',
        'api-reference/context-window',
        'api-reference/turn',
        'api-reference/memory-chunk',
        'api-reference/protocols',
      ],
    },
    {
      type: 'category',
      label: 'Performance',
      items: [
        'performance/benchmarks',
      ],
    },
    'faq',
  ],
};

export default sidebars;
