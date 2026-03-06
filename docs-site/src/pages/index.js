import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

const features = [
  {
    title: 'Metal-Accelerated Scoring',
    description:
      'Parallelized relevance and recency scoring using custom Metal compute shaders. p99 window builds in <7ms on M2.',
  },
  {
    title: 'Four-Tier Memory',
    description:
      'Working, episodic, semantic, and procedural memory tiers with automatic promotion, consolidation, and contradiction detection.',
  },
  {
    title: 'Progressive Compression',
    description:
      'Automatically applies light or heavy extractive compression to lower-signal chunks to fit your token budget. No LLM required.',
  },
  {
    title: 'Attention-Aware Reranking',
    description:
      'Re-orders context chunks based on attention centrality, placing the most critical information where the model looks first.',
  },
];

function HeroBanner() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero hero--primary" style={{padding: '4rem 0'}}>
      <div className="container" style={{textAlign: 'center'}}>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div
          style={{
            display: 'flex',
            gap: '1rem',
            justifyContent: 'center',
            marginTop: '1.5rem',
          }}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started/installation">
            Get Started
          </Link>
          <Link
            className="button button--outline button--lg"
            href="https://github.com/christopherkarani/ContextCore">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({title, description}) {
  return (
    <div className="col col--3">
      <div className="feature-card" style={{height: '100%'}}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function FeaturesSection() {
  return (
    <section style={{padding: '3rem 0'}}>
      <div className="container">
        <div className="row">
          {features.map((props, idx) => (
            <FeatureCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function PerformanceBanner() {
  return (
    <section style={{padding: '2rem 0 4rem', textAlign: 'center'}}>
      <div className="container">
        <Heading as="h2" style={{marginBottom: '1.5rem'}}>
          Performance (M2 Max)
        </Heading>
        <div
          className="row"
          style={{justifyContent: 'center', gap: '2rem'}}>
          <div
            className="feature-card"
            style={{textAlign: 'center', minWidth: '180px'}}>
            <h3 style={{fontSize: '2rem', margin: 0}}>6.54ms</h3>
            <p style={{margin: 0, opacity: 0.7}}>Window Build p99</p>
          </div>
          <div
            className="feature-card"
            style={{textAlign: 'center', minWidth: '180px'}}>
            <h3 style={{fontSize: '2rem', margin: 0}}>19.71ms</h3>
            <p style={{margin: 0, opacity: 0.7}}>Consolidation p99</p>
          </div>
          <div
            className="feature-card"
            style={{textAlign: 'center', minWidth: '180px'}}>
            <h3 style={{fontSize: '2rem', margin: 0}}>~1 MB</h3>
            <p style={{margin: 0, opacity: 0.7}}>GPU Memory</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Metal-accelerated context management for on-device AI agents.">
      <HeroBanner />
      <main>
        <FeaturesSection />
        <PerformanceBanner />
      </main>
    </Layout>
  );
}
