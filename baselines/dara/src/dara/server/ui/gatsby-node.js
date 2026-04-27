const path = require("path")

exports.createPages = async ({ actions, graphql }) => {
  const { createPage } = actions
  const mdTemplates = path.resolve(`src/templates/mdTemplates.js`)

  const result = await graphql(`
    {
      allMarkdownRemark {
        edges {
          node {
            frontmatter {
              slug
            }
          }
        }
      }
    }
  `)

  if (result.errors) {
    reporter.panicOnBuild(`Error while running GraphQL query.`)
    return
  }

  result.data.allMarkdownRemark.edges.forEach(({ node }) => {
    createPage({
      path: node.frontmatter.slug,
      component: mdTemplates,
      context: {
        slug: `${node.frontmatter.slug}`,
      }
    })
  })
}