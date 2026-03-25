import { Menu } from 'antd';
import styled from 'styled-components';
import { useState } from 'react';
import { Link } from 'gatsby';
import React from 'react';

const StyledMenu = styled(Menu)`
  border-bottom: none;
`;

function HeaderMenu() {
  let current_tag;
  if (typeof window === 'undefined') {
    current_tag = '';
  } else if (window.location.pathname === '/') {
    current_tag = 'home';  // homepage
  } else {
    current_tag = /(?<=\/)[^/]+/.exec(window.location.pathname)[0];
  }

  const [ selected_key, set_selected_key ] = useState(current_tag);

  const handleSelect = ({ key }) => {
    if (key !== 'cv' || key !== 'old-site') {
      set_selected_key(key);
    }
  };

  return (
    <StyledMenu mode='horizontal' 
      activeKey={[selected_key]}
      defaultActiveFirst={[selected_key]}
      selectedKeys={[selected_key]}
      defaultSelectedKeys={[selected_key]}
      onSelect={handleSelect}
      inlineCollapsed={false}
      disabledOverflow
    >
      <Menu.Item key='submit' title='Submit'>
        <Link to='/'>Submit</Link>
      </Menu.Item>
      <Menu.Item key='results' title='Results'>
        <Link to='/results'>Results</Link>
      </Menu.Item>
      <Menu.Item key='tutorial' title='Tutorial'>
        <Link to='/tutorial'>Tutorial</Link>
      </Menu.Item>
      <Menu.Item key='doc' title='Documentation'>
        <a href="https://CederGroupHub.github.io/dara/" target="_blank" rel="noopener noreferrer">Documentation</a>
      </Menu.Item>
    </StyledMenu>
  )
}

export default HeaderMenu;