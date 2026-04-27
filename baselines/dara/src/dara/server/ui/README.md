# Himerometra

## Installation
### Prerequisites
You should install `npm` in advance. Refer to [Official NPM Installation Guide](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for more info.

### Install develop dependencies
To set up the dev env, you need to first clone the repo to your PC and then install all the packages used in this project.
```bash
git clone https://github.com/idocx/himerometra 
cd himerometra

# use 'npm install' to install all the dependencies
npm install
```

#### Use npm mirror
For Chinese developers, you may find it slow to download the dependencies. You can use `Taobao`'s npm mirror insead.
```bash
npm config set registry https://registry.npm.taobao.org
npm config get registry
```

### Launch dev server
If you want to develop locally, you can launch a dev server with
```bash
npm run develop
```

### Build locally
We setup a github action for continuous integration. But if you want to build locally, use npm command
```bash
npm run build
```

#### Relative Path Issue
By default, the site is assumed to serve in the root path. If not, you should specify a basename manually. Himerometra can read basename from environment variable. To build, just run
```bash
PATH_PREFIX='/<Your Basename>' && npm run build --prefix-paths
```

## Add new pages
### Markdown Page
All the markdown pages are stored in [`src/md-pages`](src/md-pages).

You can just create a new `.md` file to make a new page. The new page should have following format
```
---
slug: "<the path relative to root for this page>"
hasSider: <Boolean, whether to add sider component to this page>
title: "<this will be added to the page's title>"
---

<!-- Remember: If you want to add hyperlinks to markdown file, use relative path without './' or '../'-->
<Your markdown content goes here>
```

### HTML page
You can also use html to write the page. All the html page are stored in [`src/pages`](src/pages) in `.js` file.

```js
import React from 'react;'
import Layout from '../components/Layout';

function YourCustomPage() {
  return (
    <Layout 
      hasSider={"<Boolean, whether to include sider component>"} 
      title="<the title of this page, which will be added to the page's title>"
    >
      {"<Your React content goes here>"}
    </Layout>
  )
}

export default YourCustomPage;
```

## Add new pages to header menu
To show the new page in the header menu, you should also modify the [`src/components`](src/components).

Add a `<Menu.Item />` to `<StyledMenu />` component.
```html
<StyledMenu>
  ...
  <Menu.Item key='<name>' 
    title='<Display Name>'
  >
    <Link to='<path/slug>'>Home</Link>
  </Menu.Item>
</StyledMenu>
```