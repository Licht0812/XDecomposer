import React from 'react';
import { Layout as LayoutAntd, BackTop } from 'antd';
import styled from 'styled-components';
import HeaderMenu from './HeaderMenu';
import { Link } from 'gatsby';
import Halmet from 'react-helmet';
import favicon from '../images/favicon.ico';

import 'antd/dist/antd.css';
import './layout.css';

const BREAK_POINT = 768;

const {
  Header: HeaderAntd,
  Sider: SiderAntd,
  Content: ContentAntd,
  Footer: FooterAntd,
} = LayoutAntd;

function Header({ current }) {
  return (
    <HeaderAntd>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <Link className='page-title' to='/' style={{ color: 'inherit' }}>
          <h1 className='page-title-text' 
            style={{ marginBottom: 0, fontSize: '1.5rem', fontWeight: 600 }}
          >
            Dara
          </h1>
        </Link>
        <HeaderMenu current={current}></HeaderMenu>
      </div>
    </HeaderAntd>
  )
}

function Sider() {
  return (<></>)
}

function Footer() {
  return (
      <FooterAntd
        style={{ textAlign: 'center', color: 'rgba(0, 0, 0, 0.7)', fontSize: '0.9em' }}
      >
        Powered by <a href='https://github.com/idocx/himerometra' target='_blank' rel="noreferrer">
          Himerometra</a>.
      </FooterAntd>
  )
}

const LayoutContainer = styled(LayoutAntd)`
  background: #fff;
  min-height: 100vh;

  .ant-layout-header {
    padding: 0 48px;
    margin-bottom: 16px;
    height: unset;
    border-bottom: 1px solid #f0f0f0;
  }

  .page-title-text {
    transition: all 0.2s;
  }

  .page-title-text:hover {
    color: rgba(0, 0, 0, 0.5);
  }

  .ant-layout-header,
  .ant-layout-footer {
    background: #fff;
  }

  .ant-layout-sider,
  .ant-layout-content {
    padding: 16px 24px;
  }

  .ant-layout-sider {
    transition: none;
  }

  .ant-divider-horizontal {
    margin: 8px 0;
  }

  .ant-menu-title-content {
    font-weight: 500;
  }

  .sider-container {
    display: block;
  }

  .sider-container-collapsed {
    display: none;
  }

  @media (max-width: ${BREAK_POINT}px) {
    .main-container {
      flex-direction: column !important;
    }

    .ant-layout-header {
      padding: 0 24px;
    }

    .ant-layout-sider,
    .ant-layout-content {
      padding: 4px 32px;
    }

    .ant-layout-sider {
      width: 100% !important;
      max-width: 100% !important;
      min-width: 100% !important;
      flex: unset !important;
    }

    .main-container > .ant-layout-content {
      width: 100% !important;
    }

    .sider-container {
      display: none !important;
    }
  
    .sider-container-collapsed {
      display: block !important;
    }
  }

  @media (max-width: 568px) {
    .page-title {
      display: none;
    }
  }
`;

const ContentContainer = styled(LayoutAntd)`
  background: #fff;
  flex: 1 0 auto;
`;

function Layout({ children, hasSider, title }) {
  return (
      <LayoutContainer className="layout-container">
        <Halmet>
          <title>{title ? `${title} | `: ""}Dara | Data-driven automated Rietveld analysis for phase search and refinement</title>
        </Halmet>
        <Halmet>
          <link rel="icon" href={favicon} />
        </Halmet>
        <Halmet>
          <meta charset="utf-8" />
        </Halmet>
        <Halmet>
          <meta http-equiv="X-UA-Compatible" content="chrome=1" />
        </Halmet>
        <Halmet>
          <meta name="keywords" content="Amesp, quantum chemistry, software" />
        </Halmet>
        <Halmet>
          <meta name="subject" content="product website" />
        </Halmet>

        <Header></Header>
        <BackTop />
        <ContentContainer className='main-container'>
          {hasSider ? <Sider /> : null}
          <ContentAntd>
            {children}
          </ContentAntd>
        </ContentContainer>
        <Footer></Footer>
      </LayoutContainer>
  )
}

export { BREAK_POINT };
export default Layout;