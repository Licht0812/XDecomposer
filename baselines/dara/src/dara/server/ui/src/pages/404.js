import React from 'react';
import styled from 'styled-components';
import Layout from '../components/Layout';

const NotFoundContainer = styled.div`
  font-size: 1.5rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  align-content: center;

  .not-found-code {
    font-size: 8rem;
    color: #a0a0a0;
    font-weight: 500;
  }

  .not-found-info {
    text-align: center
  }

  & code {
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 6px;
    padding: .2em .4em;
    font-size: 85%;
  }
`

function NoFoundPage() {
  const PATH_NAME = typeof window !== 'undefined' ? window.location.pathname : ''
  return (
    <Layout hasSider={false} title="Not Found">
      <NotFoundContainer className='not-found'>
        <div className='not-found-code'>
          404
        </div>
        <div className='not-found-info'>
          {PATH_NAME ? <code>{PATH_NAME}</code> : "This page"} not found in this site.
        </div>
      </NotFoundContainer>
    </Layout>
  )
}

export default NoFoundPage;