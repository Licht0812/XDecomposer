const pathPrefix = process.env.PATH_PREFIX ? process.env.PATH_PREFIX : "/";

module.exports = {
  pathPrefix: pathPrefix,
  siteMetadata: {
    title: 'Amesp - www.amesp.xyz',
  },
  plugins: [
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `md-pages`,
        path: `${__dirname}/src/md-pages/`,
      },
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        footnotes: true,
        gfm: true,
        plugins: [
          {
            resolve: `gatsby-remark-copy-linked-files`,
            options: {
              destinationDir: f => `static/${f.name}-${f.hash}`,
              ignoreFileExtensions: []
            },
          },
          {
            resolve: 'gatsby-remark-emoji',
            options: {
              emojiConversion: 'shortnameToUnicode',
              ascii: true,
            }
          },
        ],
      },
    },          
    `gatsby-plugin-react-helmet`,
    `gatsby-plugin-styled-components`,
    `gatsby-plugin-catch-links`,
    `gatsby-plugin-use-query-params`
  ]
}
